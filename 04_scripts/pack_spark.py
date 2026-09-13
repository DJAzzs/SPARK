#!/usr/bin/env python3
"""把 QAT checkpoint 打包为正式 .spark 引擎目录，并双重验证。

流程：
  1. 载入 QAT ckpt（{layer_packed, _scale_index, _meta}），直接解码写回模型
     → 生成回答（eval）
  2. 打包 .spark：state_dict.pt（与 exporter 格式同构）+ config.json（base 复制）
  3. 用 SparkWeightLoader 从 .spark 包重新加载 → 再生成回答
     → 两路回答必须完全一致（包格式无损）

用法：
    python3 04_scripts/pack_spark.py --ckpt saves/spark-qat/spark-qat-7000.pt \
        --model /home/dja/桌面/Models/Qwen2.5-0.5B-Instruct \
        --out data/spark-0.5b-qat7000
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

from block_fp2_emu import BYTES_PER_BLOCK, ELEMS_PER_BLOCK, unpack_blockwise
from spark_loader import SparkWeightLoader

PROMPTS = [
    "1+2等于几？直接回答。",
    "中国的首都是哪座城市？",
    "用一句话介绍你自己。",
    "把下面的话翻译成英文：今天天气很好。",
]


def gen_answers(model, tok, prompts, max_new_tokens):
    model.eval()
    out = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        out.append(tok.decode(o[0][inputs.input_ids.shape[1]:],
                              skip_special_tokens=True).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="QAT checkpoint (.pt)")
    ap.add_argument("--model", default="/home/dja/桌面/Models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True, help=".spark 输出目录")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    # ---- 1) 直接解码 eval ----
    print(f"[1/3] 载入 QAT ckpt 并直接解码评估: {args.ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    from spark_loader import apply_spark_state
    msd = model.state_dict()
    n = 0
    for key in list(state.keys()):
        if not key.endswith('_packed'):
            continue
        layer = key[: -len('_packed')]
        meta = state.get(layer + '_meta')
        if meta is None:
            continue
        oc, ic = int(meta[0]), int(meta[1])
        packed = state[key]
        nb = (oc * ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
        from quant_linear import _unpack_weight
        dec = _unpack_weight(packed, nb * ELEMS_PER_BLOCK).reshape(oc, -1)[:, :ic]
        sd_key = layer + '.weight'
        if sd_key in msd and msd[sd_key].shape == dec.shape:
            msd[sd_key].copy_(dec.to(msd[sd_key].dtype))
            n += 1
    print(f"      解码写回 {n} 层")
    direct_answers = gen_answers(model, tok, PROMPTS, args.max_new_tokens)

    # ---- 2) 打包 .spark ----
    print(f"[2/3] 打包 .spark -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    torch.save(state, os.path.join(args.out, "state_dict.pt"))
    src_cfg = os.path.join(args.model, "config.json")
    if os.path.exists(src_cfg):
        shutil.copy(src_cfg, os.path.join(args.out, "config.json"))
    total_mb = os.path.getsize(os.path.join(args.out, "state_dict.pt")) / 1024**2
    # 体积统计
    packed_bytes = sum(v.numel() for k, v in state.items()
                       if k.endswith('_packed'))
    print(f"      state_dict.pt {total_mb:.0f}MB (packed 权重 "
          f"{packed_bytes/1024**2:.0f}MB = {packed_bytes*8/1e9:.2f} GB-bit)")

    # ---- 3) 从 .spark 包重新加载验证 ----
    print("[3/3] 从 .spark 包加载验证（loader 路径）...")
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    loader = SparkWeightLoader(args.out)
    decoded_state = loader.state_dict()
    msd2 = model2.state_dict()
    matched = 0
    for k, w in decoded_state.items():
        if k in msd2 and msd2[k].shape == w.shape:
            with torch.no_grad():
                msd2[k].copy_(w.to(msd2[k].dtype))
            matched += 1
    print(f"      matched {matched} 层")
    spark_answers = gen_answers(model2, tok, PROMPTS, args.max_new_tokens)

    print("\n" + "=" * 70)
    print(f".spark 打包验证: {args.ckpt} -> {args.out}")
    print("=" * 70)
    all_same = True
    for p, a0, a1 in zip(PROMPTS, direct_answers, spark_answers):
        same = (a0 == a1)
        all_same &= same
        print(f"\nQ: {p}")
        print(f"  [直接解码] {a0!r}")
        print(f"  [.spark 包] {a1!r}")
        print(f"  {'>> 一致 ✓' if same else '>> 不一致 ✗'}")
    print("\n" + ("✅ .spark 包与直接解码完全一致，包格式无损可用"
                  if all_same else "❌ 两路输出不一致，需要排查"))
    print(f"包目录: {args.out}")


if __name__ == "__main__":
    main()
