#!/usr/bin/env python3
"""PTQ 硬量化评测：不做 QAT，直接量化权重，看回答效果。

流程：
  1. 加载 base 模型（CPU fp32），对一组 prompt 生成基线回答（贪心，确定性）
  2. 对所有 Linear 权重做 channel-FP2 round-trip：
     pack(候选搜索指数) -> unpack -> 写回 weight  （即"硬上量化"）
  3. 同一组 prompt 再生成，逐条对比 + 报告压缩信息

用法：
    python3 04_scripts/ptq_eval.py [--model ...] [--max-new-tokens 48]
"""
from __future__ import annotations

import argparse
import os
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

from block_fp2_emu import (ELEMS_PER_BLOCK, BYTES_PER_BLOCK,
                           pack_blockwise_search, unpack_blockwise)

PROMPTS = [
    "1+2等于几？直接回答。",
    "中国的首都是哪座城市？",
    "用一句话介绍你自己。",
    "把下面的话翻译成英文：今天天气很好。",
]


def gen_answers(model, tok, prompts, max_new_tokens):
    """贪心解码（确定性），返回每个 prompt 的回答文本。"""
    model.eval()
    answers = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True).strip()
        answers.append(ans)
        print(f"    [{time.time()-t0:5.1f}s] {p}")
    return answers


@torch.no_grad()
def ptq_harden(model, parts="all"):
    """对选定的 Linear（跳过 embed/lm_head）做 channel-FP2 round-trip 并写回权重。

    parts:
      all  - 所有 Linear（默认）
      mlp  - 只量化 MLP (gate/up/down_proj)，注意力保 fp32
      attn - 只量化注意力投影 (q/k/v/o_proj)，MLP 保 fp32
    返回 (量化层数, 量化部分 fp32 字节, packed 字节, 未量化 fp32 字节)。
    """
    n_layers = n_skipped = 0
    fp32_bytes = q_bytes = kept_fp32 = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if 'embed' in name.lower() or 'lm_head' in name:
            continue
        is_mlp = 'mlp' in name
        is_attn = ('self_attn' in name) or ('attn' in name and not is_mlp)
        if parts == "mlp" and not is_mlp:
            n_skipped += 1
            kept_fp32 += mod.weight.numel() * 4
            continue
        if parts == "attn" and not is_attn:
            n_skipped += 1
            kept_fp32 += mod.weight.numel() * 4
            continue
        w = mod.weight.data.detach().float()
        oc, ic = w.shape

        # channel pad 到 16 倍数
        nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
        ic_padded = nb_ic * ELEMS_PER_BLOCK
        if ic_padded != ic:
            w_full = torch.zeros(oc, ic_padded, dtype=w.dtype)
            w_full[:, :ic] = w
        else:
            w_full = w

        packed, _ = pack_blockwise_search(w_full)
        dec = unpack_blockwise(packed, w_full.numel(),
                               dtype=torch.float32).reshape(oc, -1)[:, :ic]

        mod.weight.data.copy_(dec.reshape_as(mod.weight.data).to(mod.weight.dtype))
        n_layers += 1
        fp32_bytes += w.numel() * 4
        q_bytes += packed.numel()
    if n_skipped:
        print(f"      [skip] {n_skipped} 层保 fp32 ({kept_fp32/1024**2:.0f}MB)")
    return n_layers, fp32_bytes, q_bytes, kept_fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/dja/桌面/Models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--parts", default="all", choices=["all", "mlp", "attn"],
                    help="all=全部 Linear | mlp=只量化 MLP | attn=只量化注意力投影")
    args = ap.parse_args()

    print(f"[1/4] 加载模型 {args.model} (CPU fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[2/4] 生成基线回答（未量化）...")
    base_answers = gen_answers(model, tok, PROMPTS, args.max_new_tokens)

    print(f"[3/4] 硬量化（parts={args.parts}，channel-FP2 round-trip 写回）...")
    t0 = time.time()
    n_layers, fp32_bytes, packed_bytes, kept_fp32 = ptq_harden(model, parts=args.parts)
    print(f"      量化 {n_layers} 层 / {time.time()-t0:.1f}s")

    print("[4/4] 生成量化后回答...")
    q_answers = gen_answers(model, tok, PROMPTS, args.max_new_tokens)

    print("\n" + "=" * 70)
    print(f"PTQ 硬量化前后对比（parts={args.parts}，贪心解码）")
    print("=" * 70)
    n_same = 0
    for p, a0, a1 in zip(PROMPTS, base_answers, q_answers):
        same = (a0 == a1)
        n_same += same
        print(f"\nQ: {p}")
        print(f"  [基线] {a0!r}")
        print(f"  [量化] {a1!r}")
        print(f"  {'>> 完全一致' if same else '>> 有变化'}")

    ratio = fp32_bytes / max(packed_bytes, 1)
    total_after = packed_bytes + kept_fp32
    total_before = fp32_bytes + kept_fp32
    print("\n" + "=" * 70)
    print(f"量化部分: fp32 {fp32_bytes/1024**2:.0f}MB -> FP2 packed "
          f"{packed_bytes/1024**2:.0f}MB  (压缩 {ratio:.1f}x, "
          f"{packed_bytes*8/max(fp32_bytes/4,1):.2f} bit/权重)")
    print(f"线性层总占用: {total_before/1024**2:.0f}MB -> {total_after/1024**2:.0f}MB "
          f"(整体压缩 {total_before/max(total_after,1):.1f}x)")
    print(f"回答完全一致率: {n_same}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
