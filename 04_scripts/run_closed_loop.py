#!/usr/bin/env python3
"""SPARK 端到端最小闭环：quantize → export(.spark) → load → generate。

当前环境（无 GPU）下走 emu 解码路径；GPU 可用时自动走 CUDA kernel。
通过限制参与量化的层数与 max_new_tokens，可在 CPU 上验证完整闭环逻辑。

用法：
    python 04_scripts/run_closed_loop.py \
        --model /home/dja/桌面/Models/Qwen2.5-0.5B-Instruct \
        --out   /tmp/spark_closed_loop \
        --max-new-tokens 8 \
        --limit-layers 1          # 仅量化前 N 层以加速 CPU 验证（默认全部）

    GPU 可用时默认全模型量化（--limit-layers 0 表示全部）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_linear import apply_channel_fp2_quant, ChannelFP2Linear
from spark_exporter import export_spark_model
from spark_loader import load_spark_model


def limit_layers(model, n: int):
    """仅让前 n 个 decoder 层的 Linear 参与量化（CPU 加速验证用）。n<=0 表示全部。

    把超出范围的 decoder 层替换为 nn.Identity 占位，apply_channel_fp2_quant
    递归遍历时它们不是 nn.Linear 因而被跳过。
    """
    if n <= 0:
        return
    from torch import nn

    # 定位 decoder layers 容器
    layers = None
    for _n in ("model.layers", "model.model.layers", "transformer.h", "model.h"):
        obj = model
        ok = True
        for part in _n.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            layers = obj
            break
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        print("[WARN] 未定位到 decoder layers，将量化全部层")
        return
    for i in range(n, len(layers)):
        layers[i] = nn.Identity()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/home/dja/桌面/Models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="/tmp/spark_closed_loop")
    ap.add_argument("--prompt", default="2+2=")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="仅量化前 N 层 (CPU 加速)；0 = 全部")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/5] 加载 base 模型 {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"      参数: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    if args.limit_layers > 0:
        limit_layers(model, args.limit_layers)
        print(f"      [限量] 仅量化前 {args.limit_layers} 层 (CPU 验证)")

    print("[2/5] 应用 ChannelFP2Linear 量化层...")
    model = apply_channel_fp2_quant(model)

    nq = 0
    print("[3/5] 逐层 quantize() (候选搜索打包)...")
    for name, module in model.named_modules():
        if isinstance(module, ChannelFP2Linear) and hasattr(module, "quantize"):
            module.quantize()
            nq += 1
    print(f"      已量化 {nq} 个 ChannelFP2 层")

    # 清空 out 目录
    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    print("[4/5] 导出 .spark ...")
    export_spark_model(model, args.out, base_model_dir=args.model)

    print("[5/5] 重新加载并生成...")
    loaded_model, _ = load_spark_model(args.out, args.model)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = loaded_model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, do_sample=False)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n=== 闭环结果 ({time.time()-t0:.1f}s) ===")
    print(result)
    print("\n✅ SPARK 端到端闭环完成")


if __name__ == "__main__":
    main()
