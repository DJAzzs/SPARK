#!/usr/bin/env python3
"""终局对比矩阵：PPL / decode 速度 / 峰值显存，多格式同台。

格式: bf16, spark(训练ckpt或v2容器), nvfp4(torchao PTQ), nf4(bnb), int8(bnb)
口径: 同一 wikitext2 数据; decode 速率用差分法((n_long-n_short)/(t_long-t_short))
      剥离 prefill; 峰值显存 torch.cuda.max_memory_allocated。
用法(GPU):
    python3 04_scripts/eval_matrix.py --model /path/to/Qwen3.5-4B \
        --spark-ckpt saves/spark-qat-4b/spark-qat-final.pt \
        --formats bf16,spark,nvfp4,nf4,int8
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_ppl import load_wikitext2_val, compute_ppl


def _load_base(model_path, device, dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, low_cpu_mem_usage=True, torch_dtype=dtype)


@torch.no_grad()
def ppl_of(model, blocks, device, max_blocks):
    model.eval()
    return compute_ppl(model, blocks, device, max_blocks)[0]


@torch.no_grad()
def speed_of(model, tok, device, n_short=8, n_long=128):
    """纯 decode 速率 (tok/s)：差分剥离 prefill；附峰值显存。"""
    import time as _t
    prompt = ("The history of artificial intelligence began in antiquity, "
              "with myths, stories and rumours of artificial beings endowed "
              "with intelligence or consciousness by master craftsmen. The "
              "seeds of modern AI were planted by philosophers who attempted "
              "to describe the process of human thinking as the mechanical "
              "manipulation of symbols.")
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    model.generate(ids, max_new_tokens=4, do_sample=False)   # 预热
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = _t.time()
    model.generate(ids, max_new_tokens=n_short, do_sample=False)
    if is_cuda: torch.cuda.synchronize()
    t1 = _t.time()
    model.generate(ids, max_new_tokens=n_long, do_sample=False)
    if is_cuda: torch.cuda.synchronize()
    t2 = _t.time()
    tps = (n_long - n_short) / max(t2 - t1, 1e-9)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if is_cuda else None
    return tps, peak


def load_model(fmt, args, dev):
    if fmt == "bf16":
        return _load_base(args.model, dev, torch.bfloat16)
    if fmt == "spark":
        assert args.spark_ckpt, "--spark-ckpt 必填"
        if os.path.isdir(args.spark_ckpt):
            from spark_v2_container import load_spark_v2
            state = load_spark_v2(args.spark_ckpt)
        else:
            state = torch.load(args.spark_ckpt, map_location="cpu",
                               weights_only=False)
        m = _load_base(args.model, "cpu", torch.float32)
        from spark_loader import apply_spark_state
        n1, n2, n3 = apply_spark_state(m, state)
        print(f"  apply: packed={n1} nvfp4={n2} params={n3}")
        return m.to(dev).to(torch.bfloat16)
    if fmt == "nvfp4":
        m = _load_base(args.model, dev, torch.bfloat16)
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
        from torchao.quantization import quantize_
        quantize_(m, NVFP4WeightOnlyConfig())
        print("  torchao NVFP4 PTQ 完成")
        return m
    if fmt == "nf4":
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(
            args.model, device_map=dev, quantization_config=cfg)
    if fmt == "int8":
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            args.model, device_map=dev, quantization_config=cfg)
    raise ValueError(f"未知格式 {fmt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spark-ckpt", default=None)
    ap.add_argument("--formats", default="bf16,spark,nvfp4")
    ap.add_argument("--data", default="data/wikitext2.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-blocks", type=int, default=40)
    args = ap.parse_args()

    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    blocks = load_wikitext2_val(tok, args.data)
    dev = args.device
    results = {}

    for fmt in fmts:
        print(f"\n===== [{fmt}] =====")
        try:
            m = load_model(fmt, args, dev)
            ppl = ppl_of(m, blocks, dev, args.max_blocks)
            tps = peak = None
            try:
                tps, peak = speed_of(m, tok, dev)
            except Exception as se:
                print(f"  [速度测试失败，PPL 保留] {type(se).__name__}: "
                      f"{str(se)[:90]}")
            results[fmt] = dict(ppl=ppl, tps=tps, peak=peak)
            extra = f" | 峰值显存 {peak:.2f}GB" if peak else ""
            sp = f"{tps:.1f} tok/s" if tps else "speed: n/a"
            print(f"  PPL={ppl:.4f} | {sp}{extra}")
            del m; gc.collect()
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}")
            results[fmt] = None

    print("\n========== 终局对比矩阵 ==========")
    hdr = f"  {'格式':8s} {'PPL':>9s} {'退化':>7s} {'tok/s':>9s} {'显存':>8s}"
    print(hdr)
    base = results.get("bf16")
    for fmt, r in results.items():
        if r is None:
            print(f"  {fmt:8s} 失败"); continue
        deg = (f"×{r['ppl']/base['ppl']:.3f}"
               if base and fmt != "bf16" else "基准")
        peak = f"{r['peak']:.2f}GB" if r.get("peak") else "-"
        tps = f"{r['tps']:.1f}" if r.get("tps") else "n/a"
        print(f"  {fmt:8s} {r['ppl']:9.3f} {deg:>7s} {tps:>9s} {peak:>8s}")


if __name__ == "__main__":
    main()
