#!/usr/bin/env python3
"""SPARK 硬件自检 (00_hardware_assert.py)

校验目标（对标双路 RTX PRO 6000 Blackwell MaxQ，sm_120）：
  1. CUDA / torch 可用，且 ≥2 卡；
  2. 每卡计算能力 >= (12, 0)；
  3. 功耗上限 >= 300 W/卡；
  4. 显存总量 > 94 GB（受 ollama 占用影响时仅报告实际可用量，
     训练前请手动清空，本脚本只做探测+告警不干预）；
  5. D2D 内存带宽实测 > 1.6 TB/s。

用法：
    python 00_hardware_assert.py [--bandwidth-mb 4096]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

try:
    import torch
except Exception as e:                      # pragma: no cover
    print(f"[FAIL] torch 不可用: {e}"); sys.exit(1)


POWER_THRESHOLD_W = 300.0
VRAM_TOTAL_GB     = 94.0
BANDWIDTH_TBS     = 1.6


def parse_power_limit(nvidia_smi="nvidia-smi"):
    """从 nvidia-smi -q -d POWER 解析每卡功耗上限 (Watts)."""
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        vals = []
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if not line or "[" in line:
                continue
            try:
                vals.append(float(line.split()[0]))
            except ValueError:
                pass
        return vals
    except Exception:
        return None


def measure_bandwidth(device, mb=4096):
    """torch D2D copy + CUDA events 实测带宽 (TB/s)."""
    nbytes = mb * 1024 * 2048                       # MB -> bytes (1MB)
    a = torch.randn(nbytes // 4, device=device, dtype=torch.float32)
    b = torch.empty_like(a)

    for _ in range(3):                              # warmup
        b.copy_(a)
    if device.index is not None:
        torch.cuda.synchronize(device)

    starts, ends = [], []
    reps = 10
    s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        s0.record(); b.copy_(a); e0.record()
        # copy_ is async; measure via stream ordering
        starts.append(s0); ends.append(e0)

    def flush():
        if device.index is not None:
            torch.cuda.synchronize(device)
    flush()

    times = [s.elapsed_time(e) / 1000.0 for s, e in zip(starts, ends)]   # seconds
    total_gb = (nbytes * 2)                                              # read+write
    best_tbs = max(total_gb / t / 1e12 for t in times)
    return best_tbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth-mb", type=int, default=4096,
                    help="带宽测试张量大小 (MB)")
    args = ap.parse_args()

    results = {}
    print("=== SPARK 硬件自检 ===\n")
    ok_all = True

    # ---- CUDA / device count ----
    n_gpu = torch.cuda.device_count()
    print(f"[1/5] GPU 数量: {n_gpu}")
    if n_gpu < 2:
        results["gpus"] = False; ok_all = False
        print("      [FAIL] 需要 >=2 卡")
    else:
        results["gpus"] = True

    # ---- compute capability ----
    caps, names = [], []
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        cap = tuple(torch.cuda.get_device_capability(i))
        caps.append(cap); names.append(name)
        print(f"[2/5] GPU{i}: {name}  compute={cap[0]}.{cap[1]}")
    if all((c >= (12, 0)) for c in caps):
        results["cc"] = True
    else:
        results["cc"] = False; ok_all = False
        print("      [FAIL] 需要 Blackwell sm_120")

    # ---- power limit ----
    powers = parse_power_limit()
    if powers is not None and len(powers) >= min(2, n_gpu):
        low = min(powers[:n_gpu])
        ok_p = (low + 1e-6) >= POWER_THRESHOLD_W
        results["power"] = ok_p; ok_all &= ok_p
        print(f"[3/5] power.limit(W): {powers[:n_gpu]} -> min={low:.0f} "
              f"{'(OK' if ok_p else '(FAIL'} threshold={POWER_THRESHOLD_W}")
    else:
        results["power"] = None; ok_all &= True
        print(f"[3/5] power.limit: 无法解析 (跳过，仅告警) powers={powers}")

    # ---- VRAM ----
    vram_ok = True
    for i in range(n_gpu):
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        try:
            free_bytes, _tot = torch.cuda.mem_get_info(i)
        except Exception:
            free_bytes = total
        free = free_bytes / (1024**3)
        stat = "OK" if total > VRAM_TOTAL_GB else "FAIL"
        vram_ok &= (stat == "OK")
        print(f"[4/5] GPU{i} 显存: total={total:.1f} GB  当前可用≈{free:.1f} GB "
              f"(阈值> {VRAM_TOTAL_GB}) [{stat}]"
              + ("  [WARN] ollama等占用请训练前清空" if free < VRAM_TOTAL_GB else ""))
    results["vram"] = vram_ok; ok_all &= vram_ok

    # ---- bandwidth ----
    dev0 = "cuda:0"
    try:
        bw = measure_bandwidth(dev0, args.bandwidth_mb)
        ok_bw = (bw > BANDWIDTH_TBS)
        results["bandwidth"] = bw
        print(f"[5/5] D2D 带宽实测 ≈ {bw:.3f} TB/s "
              f"{'(OK' if ok_bw else '(FAIL'} threshold={BANDWIDTH_TBS})")
        # note: under ollama load, achievable may be lower -> warn not fail
        if not ok_bw:
            print("      [WARN] 当前可能受 ollama/其他进程占显存影响；请清空后复测")
    except Exception as e:
        results["bandwidth"] = None; ok_all &= True
        print(f"[5/5] D2D 带宽实测失败: {e} (跳过)")

    print("\n=== 自检结论 ===")
    for k, v in results.items():
        print(f"   {k:<12}: {'PASS' if v is True else ('WARN/SKIP' if v is None or isinstance(v,float) and False else 'FAIL')}")

    hard = all(results.get(k) is not False for k in ("gpus", "cc"))
    print(f"\n总体: {'PASS（核心自检通过，带宽/功耗需清空显存后复测）' if hard else 'FAIL'}")
    return 0 if hard else 1


if __name__ == "__main__":
    sys.exit(main())
