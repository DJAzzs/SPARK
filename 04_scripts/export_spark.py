#!/usr/bin/env python3
"""SPARK v2.3 导出：QAT SPFP2 MLP + PTQ FP3/NVFP4 attention → .spark 容器

流程:
  1. 加载 QAT ckpt (SPFP2 packed MLP + param:: 训练后权重)
  2. 从 param:: 提取训练后的 attention/out_proj/embed 权重
  3. 对 attention/GDN 施加 FP3 PTQ (从训练后权重，非 base 权重!)
  4. 对 out_proj/embed 施加 NVFP4 PTQ
  5. 打包 .spark v2 容器 + PPL 验证 + 生成测试

用法:
    python3 04_scripts/export_spark.py \
        --ckpt saves/spark-4b-const/spark-qat-final.pt \
        --model /path/to/Qwen3.5-4B \
        --out data/spark-4b-engine
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "02_model"),
           os.path.join(_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_spark(ckpt_path, model_path, out_dir, device="cuda:0"):
    from block_fp2_emu import channel_fake_quantize
    from fp3_emu import fp3_fake_quant, fp3_pack_3bit
    from nvfp4_emu import nvfp4_fake_quant, nvfp4_pack

    print(f"[1/5] 加载 ckpt: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_packed = sum(1 for k in state if k.endswith('_packed'))
    n_params = sum(1 for k in state if k.startswith('param::'))
    print(f"  SPFP2={n_packed} params={n_params}")

    print(f"[2/5] 构建导出 state（SPFP2 from ckpt + PTQ attention）")
    export_state = {}

    # SPFP2: 直接从 ckpt 搬运（QAT 训练过的）
    for k, v in state.items():
        if k.endswith('_packed'):
            export_state[k] = v
            layer = k[:-len('_packed')]
            meta = state.get(layer + '_meta')
            if meta is not None:
                export_state[layer + '_meta'] = meta

    # PTQ attention/GDN: 从训练后的 param:: 权重施加 FP3/NVFP4
    n_fp3 = n_nvfp4 = 0
    for k, v in state.items():
        if not k.startswith('param::'):
            continue
        pname = k[len('param::'):]
        # 跳过非 Linear 权重（norm, A_log, dt_bias, conv1d 等直接保留）
        if not pname.endswith('.weight'):
            export_state[k] = v
            continue
        if 'mlp' in pname:
            continue  # MLP 已由 SPFP2 packed 覆盖
        if v.dim() != 2:
            export_state[k] = v  # conv1d(3D) / 其他非 2D 直接保留
            continue

        # 训练后的 BF16 attention 权重
        w = v.float()

        if 'out_proj' in pname or 'embed' in pname or 'lm_head' in pname:
            # NVFP4 PTQ
            wq = nvfp4_fake_quant(w)
            codes, scales = nvfp4_pack(wq)
            layer = pname[:-len('.weight')]
            export_state[layer + '_nvfp4_codes'] = codes
            export_state[layer + '_nvfp4_scales'] = scales
            export_state[layer + '_meta'] = torch.tensor(list(w.shape[:2]))
            n_nvfp4 += 1
        else:
            # FP3 PTQ（attention/GDN 投影）
            wq = fp3_fake_quant(w)
            codes, scales = fp3_pack_3bit(wq)
            layer = pname[:-len('.weight')]
            export_state[layer + '_fp3_codes'] = codes
            export_state[layer + '_fp3_scales'] = scales
            export_state[layer + '_meta'] = torch.tensor(list(w.shape[:2]))
            n_fp3 += 1

    print(f"  SPFP2={n_packed} FP3_PTQ={n_fp3} NVFP4_PTQ={n_nvfp4} "
          f"params={sum(1 for k in export_state if k.startswith('param::'))}")

    print(f"[3/5] 打包 .spark v2 → {out_dir}")
    from spark_v2_container import save_spark_v2, load_spark_v2, container_size_gb
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    stats = save_spark_v2(export_state, out_dir, base_model_dir=model_path)
    gb = container_size_gb(out_dir)
    print(f"  raw {stats['raw']/1024**3:.2f}GB → 容器 {gb:.3f}GB")

    print(f"[4/5] Roundtrip 验证...")
    state2 = load_spark_v2(out_dir)
    ok_p = sum(1 for k in state2 if k.endswith('_packed'))
    ok_f = sum(1 for k in state2 if k.endswith('_fp3_codes'))
    ok_n = sum(1 for k in state2 if k.endswith('_nvfp4_codes'))
    print(f"  回读: SPFP2={ok_p}/{n_packed} FP3={ok_f}/{n_fp3} NVFP4={ok_n}/{n_nvfp4}")
    assert ok_p == n_packed and ok_f == n_fp3 and ok_n == n_nvfp4
    print(f"  ✅ 层数匹配")

    print(f"[5/5] PPL 验证...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(model_path,
        device_map=device, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    from spark_loader import apply_spark_state
    n1, n2, n3 = apply_spark_state(m, state2)
    m.eval()
    print(f"  apply: packed={n1} nvfp4(fp3+nvfp4)={n2} params={n3}")

    # PPL (前 20 块，快速)
    from eval_ppl import load_wikitext2_val
    blocks = load_wikitext2_val(tok, os.path.join(_ROOT, 'data', 'wikitext2.txt'))
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, 20, 4):
            ids = blocks[i:i+4].to(device)
            with torch.amp.autocast(device.split(':')[0], dtype=torch.bfloat16):
                out = m(input_ids=ids, labels=ids)
            total_nll += out.loss.item() * ids.numel()
            total_tok += ids.numel()
    ppl = math.exp(total_nll / total_tok)
    print(f"  导出引擎 PPL (20块): {ppl:.2f}")
    print(f"\n{'='*60}")
    print(f"  ✅ SPARK v2.3 引擎: {out_dir}")
    print(f"     体积: {gb:.3f} GB | PPL: {ppl:.2f} | 基线: 8.88")
    print(f"     退化: ×{ppl/8.88:.2f}")

    return gb, ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    export_spark(args.ckpt, args.model, args.out, args.device)


if __name__ == "__main__":
    main()
