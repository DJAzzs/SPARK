#!/usr/bin/env python3
"""WikiText-2 验证集 PPL 评测：量化模型 vs 基线的困惑度退化。

用法：
  基线 PPL:
      python3 04_scripts/eval_ppl.py --model /path/to/base
  QAT checkpoint PPL（含基线对照）:
      python3 04_scripts/eval_ppl.py --model /path/to/base --ckpt saves/spark-qat/spark-qat-4000.pt

数据源优先级：本地 data/wikitext2-val (save_to_disk) → HF 在线下载。
PPL 口径：验证集全文拼接 → 2048 token 块 → exp(总NLL/总token)（标准做法，
与 GPTQ/AWQ 论文的 WikiText-2 PPL 可比）。
"""
from __future__ import annotations

import argparse
import math
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

BLOCK = 2048


def load_wikitext2_val(tokenizer, data_dir=None):
    """加载验证集并切成 2048 token 块。默认 WikiText-2 raw val。"""
    texts = None
    local = data_dir or os.path.join(_PROJECT_ROOT, "data", "wikitext2-val")
    if local and local.endswith(".txt") and os.path.isfile(local):
        with open(local, encoding="utf-8") as f:
            texts = [line for line in f if line.strip()]
    try:
        if texts is None and os.path.isdir(local):
            from datasets import load_from_disk
            ds = load_from_disk(local)
            texts = [t for t in ds["text"] if t.strip()]
    except Exception as e:
        print(f"[WARN] 本地 wikitext 读取失败: {e}")
    if texts is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            texts = [t for t in ds["text"] if t.strip()]
        except Exception as e:
            raise SystemExit(
                f"[ERROR] WikiText-2 不可用（本地 {local} 不存在且在线下载失败: {e}）。\n"
                f"宿主机执行: python3 -c \"from datasets import load_dataset; "
                f"load_dataset('wikitext','wikitext-2-raw-v1',split='validation')"
                f".save_to_disk('data/wikitext2-val')\"")

    full = "\n\n".join(texts + [""])           # 拼接保留段落边界
    ids = tokenizer(full, return_tensors="pt").input_ids[0]
    n_blocks = ids.numel() // BLOCK
    blocks = ids[:n_blocks * BLOCK].view(n_blocks, BLOCK)
    print(f"[PPL] WikiText-2 val: {len(texts)} 行 -> {ids.numel()} tok -> "
          f"{n_blocks} 块 × {BLOCK}")
    return blocks


@torch.no_grad()
def compute_ppl(model, blocks, device, max_blocks=40, batch=4):
    """标准 PPL：exp(总 NLL / 总 token)。"""
    model.eval()
    blocks = blocks[:max_blocks].to(device)
    total_nll, total_tok = 0.0, 0
    for i in range(0, blocks.shape[0], batch):
        ids = blocks[i:i + batch]
        out = model(input_ids=ids, labels=ids)
        # HF loss = mean NLL over batch tokens；换算总 NLL
        n_tok = ids.numel()
        total_nll += out.loss.item() * n_tok
        total_tok += n_tok
    return math.exp(total_nll / total_tok), total_tok


def load_qat_weights(model, ckpt_path):
    """QAT checkpoint 的 packed 权重解码写回（v1/v2 自检测）。"""
    from quant_linear import _unpack_weight, ELEMS_PER_BLOCK
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
        dec = _unpack_weight(state[key], oc * nb_ic * ELEMS_PER_BLOCK)
        sd_key = layer + '.weight'
        if sd_key in msd and msd[sd_key].shape == dec.reshape(oc, -1)[:, :ic].shape:
            msd[sd_key].copy_(dec.reshape(oc, -1)[:, :ic].to(msd[sd_key].dtype))
            n += 1
    print(f"[PPL] ckpt 解码写回 {n} 层")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None, help="QAT checkpoint（不传则只测基线）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-blocks", type=int, default=40,
                    help="评测块数（40×2048=8.2万 token，PPL 已稳定）")
    ap.add_argument("--data", default=None,
                    help="本地数据集目录 (save_to_disk 格式, 含 text 列)；"
                         "默认 data/wikitext2-val")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    blocks = load_wikitext2_val(tok, args.data)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=args.device, low_cpu_mem_usage=True,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16)

    ppl_base, ntok = compute_ppl(model, blocks, args.device, args.max_blocks)
    print(f"[PPL] 基线 (未量化): {ppl_base:.3f}  ({ntok} tok)")

    if args.ckpt:
        load_qat_weights(model, args.ckpt)
        ppl_q, _ = compute_ppl(model, blocks, args.device, args.max_blocks)
        ratio = ppl_q / ppl_base
        print(f"[PPL] 量化后 ({os.path.basename(args.ckpt)}): {ppl_q:.3f}")
        print(f"[PPL] 退化: ×{ratio:.3f}  "
              f"({'优秀(<1.05)' if ratio < 1.05 else '良好(<1.2)' if ratio < 1.2 else '可接受(<2)' if ratio < 2 else '严重退化'})")


if __name__ == "__main__":
    main()
