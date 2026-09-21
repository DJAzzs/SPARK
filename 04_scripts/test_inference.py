#!/usr/bin/env python3
"""SPARK 量化模型推理测试 + .spark v2 打包

用法（GPU）:
    # 测试 1000 步 ckpt 的回答质量
    python3 04_scripts/test_inference.py \
        --ckpt saves/spark-4b-exp1/spark-qat-1000.pt \
        --model /home/dja/桌面/Models/Qwen3.5-4B \
        --interactive

    # 打包 .spark v2 引擎
    python3 04_scripts/test_inference.py \
        --ckpt saves/spark-4b-exp1/spark-qat-1000.pt \
        --model /home/dja/桌面/Models/Qwen3.5-4B \
        --pack data/spark-4b-exp1.spark
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from spark_loader import apply_spark_state

PROMPTS = [
    "1+2等于几？直接回答。",
    "中国的首都是哪座城市？",
    "用一句话介绍你自己。",
    "把下面的话翻译成英文：今天天气很好。",
    "Write a haiku about autumn leaves.",
    "What is 15 * 23?",
    "简要说明什么是量子纠缠。",
    "List three programming languages and their main use cases.",
]


def generate(model, tok, prompt, max_new=128, device="cuda:0"):
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", default=None, help="打包 .spark v2 输出目录")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 加载 + 应用 ckpt ----
    print(f"[1/3] 加载 base 模型 + 应用 ckpt: {args.ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16)
    if os.path.isdir(args.ckpt):
        from spark_v2_container import load_spark_v2
        state = load_spark_v2(args.ckpt)
    else:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n1, n2, n3 = apply_spark_state(model, state)
    print(f"      apply: packed={n1} nvfp4={n2} params={n3}")
    model.eval()
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.startswith("cuda") else 0

    # ---- 生成测试 ----
    print(f"\n[2/3] 回答质量测试 (device={device}, 峰值显存={peak:.2f}GB):")
    print("=" * 70)
    total_tokens, total_time = 0, 0
    for prompt in PROMPTS:
        ans, n_new, dt = generate(model, tok, prompt, args.max_new, device)
        tps = n_new / dt if dt > 0 else 0
        total_tokens += n_new
        total_time += dt
        print(f"\nQ: {prompt}")
        print(f"A: {ans}")
        print(f"   [{n_new} tokens in {dt:.1f}s = {tps:.1f} tok/s]")
    print(f"\n{'='*70}")
    print(f"总计: {total_tokens} tokens / {total_time:.1f}s "
          f"= {total_tokens/total_time:.1f} tok/s")

    # ---- 交互模式 ----
    if args.interactive:
        print("\n[交互模式] 输入问题（quit 退出）:")
        while True:
            try:
                q = input("\nYou: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                ans, n, dt = generate(model, tok, q, args.max_new, device)
                print(f"SPARK: {ans}")
                print(f"       [{n} tok / {dt:.1f}s = {n/dt:.1f} tok/s]")
            except (EOFError, KeyboardInterrupt):
                break

    # ---- 打包 ----
    if args.pack:
        print(f"\n[3/3] 打包 .spark v2 → {args.pack}")
        from spark_v2_container import save_spark_v2, container_size_gb
        import shutil
        if os.path.exists(args.pack):
            shutil.rmtree(args.pack)
        stats = save_spark_v2(state, args.pack, base_model_dir=args.model)
        gb = container_size_gb(args.pack)
        print(f"      raw {stats['raw']/1024**3:.2f}GB → 容器 {gb:.3f}GB")
        print(f"      ✅ .spark v2 引擎: {args.pack}")


if __name__ == "__main__":
    main()
