#!/usr/bin/env python3
"""评估 QAT checkpoint：解码权重 + 回答质量对比 + 量化健康度统计。

用法：
    python3 04_scripts/eval_qat_ckpt.py --ckpt saves/spark-qat/spark-qat-1000.pt
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

from block_fp2_emu import BYTES_PER_BLOCK, ELEMS_PER_BLOCK, unpack_blockwise

PROMPTS = [
    "1+2等于几？直接回答。",
    "中国的首都是哪座城市？",
    "用一句话介绍你自己。",
    "把下面的话翻译成英文：今天天气很好。",
]


def gen_answers(model, tok, prompts, max_new_tokens):
    model.eval()
    answers = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True).strip()
        answers.append(ans)
        print(f"    [{time.time()-t0:5.1f}s] {p}")
    return answers


@torch.no_grad()
def load_qat_weights(model, ckpt_path):
    """把 QAT checkpoint 的 packed 权重解码写回 model。返回统计信息。"""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msd = model.state_dict()

    n_layers = 0
    packed_bytes = 0
    expv_all = []
    n_matched = 0
    rel_errs = []
    for key in list(state.keys()):
        if not key.endswith('_packed'):
            continue
        layer = key[: -len('_packed')]
        packed = state[key]
        meta = state.get(layer + '_meta')
        if meta is None:
            continue
        oc, ic = int(meta[0]), int(meta[1])
        n_blocks = packed.numel() // BYTES_PER_BLOCK
        dec = unpack_blockwise(packed, n_blocks * ELEMS_PER_BLOCK,
                               dtype=torch.float32).reshape(oc, -1)[:, :ic]
        # 写回
        sd_key = layer + '.weight'
        if sd_key in msd and msd[sd_key].shape == dec.shape:
            w_orig = msd[sd_key].float()
            rel = ((w_orig - dec) ** 2).sum().sqrt() / \
                (w_orig ** 2).sum().sqrt().clamp_min(1e-6)
            rel_errs.append(rel.item())
            msd[sd_key].copy_(dec.to(msd[sd_key].dtype))
            n_matched += 1
        n_layers += 1
        packed_bytes += packed.numel()
        si = state.get(layer + '_scale_index')
        if si is not None:
            expv_all.append(si.long())

    stats = {
        "n_layers": n_layers,
        "n_matched": n_matched,
        "packed_MB": packed_bytes / 1024**2,
        "rel_err_mean": sum(rel_errs) / max(len(rel_errs), 1),
        "rel_err_max": max(rel_errs) if rel_errs else 0.0,
    }
    if expv_all:
        hist = torch.bincount(torch.cat(expv_all), minlength=16).tolist()
        stats["exp_hist"] = hist
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="saves/spark-qat/spark-qat-1000.pt")
    ap.add_argument("--model", default="/home/dja/桌面/Models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--skip-base", action="store_true",
                    help="跳过基线生成（已有记录时）")
    args = ap.parse_args()

    print(f"[1/3] 加载 base 模型 {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_answers = None
    if not args.skip_base:
        print("[2/3] 生成基线回答（未量化）...")
        base_answers = gen_answers(model, tok, PROMPTS, args.max_new_tokens)

    print(f"[3/3] 加载 QAT checkpoint {args.ckpt} ...")
    stats = load_qat_weights(model, args.ckpt)
    print(f"      解码 {stats['n_matched']}/{stats['n_layers']} 层写回, "
          f"packed {stats['packed_MB']:.0f}MB")
    print(f"      权重相对误差(QAT后 vs 原始): mean={stats['rel_err_mean']:.3f} "
          f"max={stats['rel_err_max']:.3f}")
    if "exp_hist" in stats:
        print("      块指数直方图 (0..15):")
        for e, c in enumerate(stats["exp_hist"]):
            if c:
                print(f"        e={e:2d} ({2.0**(e-10):.4f}): {'#'*min(60, c//500)} {c}")

    qat_answers = gen_answers(model, tok, PROMPTS, args.max_new_tokens)

    print("\n" + "=" * 70)
    print(f"QAT checkpoint 回答质量（贪心解码）: {args.ckpt}")
    print("=" * 70)
    for i, p in enumerate(PROMPTS):
        print(f"\nQ: {p}")
        if base_answers:
            print(f"  [基线   ] {base_answers[i]!r}")
        print(f"  [QAT-ckpt] {qat_answers[i]!r}")


if __name__ == "__main__":
    main()
