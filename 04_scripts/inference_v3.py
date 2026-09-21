#!/usr/bin/env python3
"""SPARK v3 推理引擎 — 码流驻留 VRAM（<2GB 目标）+ kernel 在线解码

对比:
  v2 路径: .spark → 解码为 BF16 → 物化 8 GB → F.linear → ~55 tok/s
  v3 路径: .spark → 码流驻留 ~1.5 GB → kernel 解码 → 目标 >100 tok/s

用法:
    python3 04_scripts/inference_v3.py \
        --spark data/spark-4b-v23 \
        --model /path/to/Qwen3.5-4B \
        --interactive
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class V3SPFP2Layer(nn.Module):
    """SPFP2 码流驻留 + CUDA kernel 在线解码（不物化 BF16）。"""

    def __init__(self, packed_weight, out_features, in_features, bias=None):
        super().__init__()
        self.register_buffer('w_packed', packed_weight)
        self.out_features = out_features
        self.in_features = in_features
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.register_buffer('bias', torch.zeros(0))

    def forward(self, x):
        from build_spark_v3 import load_v3_kernel
        m = load_v3_kernel()
        # 确保输入是 half
        if x.dtype != torch.float16:
            x = x.half()
        y = torch.zeros(x.shape[0], self.out_features,
                       dtype=torch.float16, device=x.device)
        m.spark_v3_spfp2_forward(
            self.w_packed, x, y, self.bias,
            self.out_features, self.in_features)
        return y.to(torch.bfloat16)


def build_v3_model(spark_dir, model_path, device="cuda:0"):
    """从 .spark 容器构建 v3 推理模型（码流驻留）。"""
    from spark_v2_container import load_spark_v2

    print(f"[v3] 加载 .spark: {spark_dir}")
    state = load_spark_v2(spark_dir)
    n_packed = sum(1 for k in state if k.endswith('_packed'))
    print(f"  SPFP2={n_packed} 层")

    # 加载 base 模型（用于非量化层的结构和权重）
    print(f"[v3] 加载 base 模型...")
    m = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16)

    # 获取 state dict 引用
    msd = m.state_dict()

    # 替换 SPFP2 层为 v3 kernel 层
    print(f"[v3] 替换 MLP 层为 kernel 推理...")
    n_replaced = 0
    for name, module in list(m.named_modules()):
        if 'mlp' not in name or not isinstance(module, nn.Linear):
            continue
        # 找对应的 packed 权重
        packed_key = name + '_packed'
        if packed_key not in state:
            continue

        parts = name.split('.')
        parent_path = '.'.join(parts[:-1])
        leaf = parts[-1]
        parent = m.get_submodule(parent_path)

        # 创建 v3 层（码流驻留）
        v3_layer = V3SPFP2Layer(
            state[packed_key].to(device),
            module.out_features, module.in_features,
            state.get(f'param::{name}.bias', torch.zeros(0)).to(device)
            if f'param::{name}.bias' in state else None)
        setattr(parent, leaf, v3_layer)
        n_replaced += 1

    print(f"  替换 {n_replaced} 层 → v3 kernel")
    m.eval()
    return m


def generate_v3(model, tok, prompt, max_new=128, device="cuda:0"):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    inp = {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()
           if k in ("input_ids", "attention_mask")}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    n_new = out.shape[1] - inp["input_ids"].shape[1]
    ans = tok.decode(out[0][inp["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    return ans, n_new, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True, help=".spark v2 容器目录")
    ap.add_argument("--model", required=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    m = build_v3_model(args.spark, args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 显存统计
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n[v3] 峰值显存: {peak:.2f} GB ← 目标 <2 GB")
    print(f"[v3] 对比 BF16 物化: ~8 GB")

    # 测试
    prompts = [
        "1+2等于几？直接回答。",
        "What is 15 * 23?",
        "中国的首都是哪座城市？",
    ]
    print(f"\n{'='*60}")
    for prompt in prompts:
        ans, n, dt = generate_v3(m, tok, prompt, 64, device)
        tps = n / dt if dt > 0 else 0
        print(f"\nQ: {prompt}")
        print(f"A: {ans[:100]}")
        print(f"   [{n} tok / {dt:.1f}s = {tps:.1f} tok/s]")

    if args.interactive:
        print(f"\n{'='*60}")
        print("交互模式 (quit 退出):")
        while True:
            try:
                q = input("\nYou: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                ans, n, dt = generate_v3(m, tok, q, 128, device)
                print(f"SPARK: {ans}")
                print(f"       [{n} tok / {dt:.1f}s = {n/dt:.1f} tok/s]")
            except (EOFError, KeyboardInterrupt):
                break

    final_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n[v3] 最终峰值显存: {final_peak:.2f} GB")


if __name__ == "__main__":
    main()
