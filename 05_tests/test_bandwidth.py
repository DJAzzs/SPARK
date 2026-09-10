"""SPARK 内存带宽实测 (05_tests/test_bandwidth.py)

用 torch D2D copy + CUDA events，测量设备到设备（read+write）有效带宽。
目标：> 1.6 TB/s（对标 1.8TB/s 满血 HBM3e）。

用法：
    python 05_tests/test_bandwidth.py [--mb 4096]
"""
from __future__ import annotations

import argparse
import sys
import time  # noqa: F401 (unused placeholder)

import torch


def measure_best(device="cuda:0", mb=4096, reps=20):
    nbytes = int(mb * (1024 ** 2))
    a = torch.randn(nbytes // 4, device=device, dtype=torch.float32)
    b = torch.empty_like(a)

    for _ in range(5):                              # warmup
        b.copy_(a)

    import sys as _sys
    s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
    real = []
    for _ in range(reps):
        s0.record(); b.copy_(a); e0.record()
        torch.cuda.synchronize(device)
        t_ms = s0.elapsed_time(e0) / 1000.0
        if t_ms > 1e-6:
            real.append(t_ms)

    best_t = min(t for t in real if t > 1e-6)
    gb_moved = (nbytes * 2)                          # read a + write b
    bw = gb_moved / best_t                           # bytes/s
    return bw / 1e12                                 # TB/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=4096)
    args = ap.parse_args()

    torch.cuda.init()
    dev = "cuda:0"
    name = torch.cuda.get_device_name(0)
    print(f"=== 带宽测试 {name} ({args.mb} MB) ===")

    best_rep = float("inf")
    for attempt in range(3):
        t = measure_best(dev, args.mb)
        if t < best_rep:
            best_rep = t
    bw_tbs = best_rep
    print(f"   D2D 带宽 ≈ {bw_tbs:.4f} TB/s")
    ok = bw_tbs > 1.6
    print("   结果:", "PASS (>=1.6TB/s)" if ok else
          "[WARN] <1.6TB/s，可能受显存占用影响；清空后复测")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
