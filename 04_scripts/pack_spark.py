#!/usr/bin/env python3
"""把 QAT checkpoint 打包为正式 .spark v2 引擎，并双重验证。

流程：
  1. 载入 ckpt（.pt 或 v2 容器），apply_spark_state 应用 → 生成回答（eval）
  2. 导出 v2 容器（NVFP4 原生码流 + SPFP2 zstd + INT8 head + fp32 参数）
  3. 从 v2 容器重新加载 → 再生成 → 双路必须一致

用法：
    python3 04_scripts/pack_spark.py \
        --ckpt saves/spark-qat-4b/spark-qat-final.pt \
        --model /home/dja/桌面/Models/Qwen3.5-4B \
        --out data/spark-4b-v2
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spark_loader import apply_spark_state
from spark_v2_container import save_spark_v2, load_spark_v2, container_size_gb

PROMPTS = [
    "1+2等于几？直接回答。",
    "中国的首都是哪座城市？",
    "用一句话介绍你自己。",
    "把下面的话翻译成英文：今天天气很好。",
]


def gen_answers(model, tok, prompts, max_new_tokens):
    model.eval()
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        inp = {k: v for k, v in tok(text, return_tensors="pt").items()
               if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(o[0][inp["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip())
    return outs


def _load_state(ckpt):
    if os.path.isdir(ckpt):
        return load_spark_v2(ckpt)
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="QAT ckpt (.pt) 或 v2 容器目录")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help=".spark v2 输出目录")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 1) ckpt 直接应用 + eval ----
    print(f"[1/3] 载入并应用 ckpt: {args.ckpt}")
    state = _load_state(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    n1, n2, n3 = apply_spark_state(model, state)
    print(f"      apply: packed={n1} nvfp4={n2} params={n3}")
    direct = gen_answers(model, tok, PROMPTS, args.max_new_tokens)
    del model

    # ---- 2) 导出 v2 容器 ----
    print(f"[2/3] 导出 .spark v2 -> {args.out}")
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    stats = save_spark_v2(state, args.out, base_model_dir=args.model)
    gb = container_size_gb(args.out)
    n_params = sum(v.numel() for k, v in state.items()
                   if not k.endswith(("_meta", "_scale_index")))
    print(f"      raw {stats['raw']/1024**3:.2f}GB → 容器 {gb:.3f}GB "
          f"(≈ {gb*8/max(n_params,1):.2f} bit/参数等效)")

    # ---- 3) v2 容器加载验证 ----
    print("[3/3] 从 v2 容器重新加载验证...")
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    state2 = load_spark_v2(args.out)
    m1, m2, m3 = apply_spark_state(model2, state2)
    print(f"      apply: packed={m1} nvfp4={m2} params={m3}")
    spark_ans = gen_answers(model2, tok, PROMPTS, args.max_new_tokens)

    print("\n" + "=" * 66)
    all_same = True
    for p, a, b in zip(PROMPTS, direct, spark_ans):
        same = (a == b)
        all_same &= same
        print(f"Q: {p}")
        print(f"  [ckpt ] {a[:64]!r}")
        if not same:
            print(f"  [v2   ] {b[:64]!r}  ✗ 不一致")
    print(("✅ v2 容器与 ckpt 完全一致 | " if all_same else "❌ 不一致 | ")
          + f"引擎: {args.out} ({gb:.3f}GB)")


if __name__ == "__main__":
    main()
