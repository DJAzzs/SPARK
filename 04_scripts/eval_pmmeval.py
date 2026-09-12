#!/usr/bin/env python3
"""P-MMEval 最终验收：选择题 loglikelihood 评测（量化模型 vs 基线）。

任务（本目录 Qwen--P-MMEval，按语言 jsonl）：
  - mmmlu  : {Question, A, B, C, D, Answer}  4 选 1（多语言 MMLU）
  - xnli   : {premise, statement, answer}    3 选 1（A=entail B=neutral C=contradiction）
评分：对每个选项构造 context+option 序列，取 option 片段 loglikelihood 之和，
argmax 与标准答案比对（与 lm-eval-harness 的 loglikelihood 口径一致，无生成）。

用法：
  python3 04_scripts/eval_pmmeval.py --model /path/to/base \
      [--ckpt saves/spark-qat/spark-qat-final.pt] [--langs zh,en] [--limit 200]
"""
from __future__ import annotations

import argparse
import json
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

MM_ROOT = os.path.join(_PROJECT_ROOT, "Qwen--P-MMEval")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_mmmlu_tasks(langs, limit, split="val"):
    tasks = {}
    for lang in langs:
        p = os.path.join(MM_ROOT, "mmmlu", split, f"{lang}.jsonl")
        if os.path.isfile(p):
            rows = load_jsonl(p)[:limit]
            tasks[f"mmmlu/{lang}"] = [
                {"ctx": r["Question"], "opts": [r["A"], r["B"], r["C"], r["D"]],
                 "gold": "ABCD".index(r["Answer"])} for r in rows]
    return tasks


def build_hellaswag_tasks(langs, limit, split):
    tasks = {}
    for lang in langs:
        p = os.path.join(MM_ROOT, "mhellaswag", split, f"{lang}.jsonl")
        if os.path.isfile(p):
            rows = load_jsonl(p)[:limit]
            tasks[f"mhellaswag/{lang}"] = [
                {"ctx": r["ctx"], "opts": r["endings"], "gold": int(r["label"])}
                for r in rows]
    return tasks


def build_logiqa_tasks(langs, limit, split):
    tasks = {}
    for lang in langs:
        p = os.path.join(MM_ROOT, "mlogiqa", split, f"{lang}.jsonl")
        if os.path.isfile(p):
            rows = load_jsonl(p)[:limit]
            tasks[f"mlogiqa/{lang}"] = [
                {"ctx": f"{r['context']}\n{r['question']}",
                 "opts": r["options"], "gold": int(r["answer"])}
                for r in rows]
    return tasks


def build_xnli_tasks(langs, limit, split):
    tasks = {}
    for lang in langs:
        p = os.path.join(MM_ROOT, "xnli", split, f"{lang}.jsonl")
        if os.path.isfile(p):
            rows = load_jsonl(p)[:limit]
            tasks[f"xnli/{lang}"] = [
                {"ctx": f"{r['premise']}\n{r['statement']}",
                 "opts": ["正确 (entailment)", "无关 (neutral)", "矛盾 (contradiction)"],
                 "gold": "ABC".index(r["answer"])} for r in rows]
    return tasks


@torch.no_grad()
def option_loglikelihood(model, tok, ctx, opt, device):
    """log P(opt | ctx)：option 片段的 loglikelihood 之和。"""
    ctx_ids = tok(ctx, return_tensors="pt").input_ids
    full_ids = tok(ctx + " " + opt, return_tensors="pt").input_ids
    n_ctx = ctx_ids.shape[1]
    n_opt = full_ids.shape[1] - n_ctx
    if n_opt <= 0:
        return -1e9
    ids = full_ids.to(device)
    logits = model(input_ids=ids).logits[0]             # [L, V]
    # 预测第 i 个 token 的分布在位置 i-1
    logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[0, 1:]
    tok_ll = logprobs[torch.arange(len(tgt)), tgt]       # [L-1]
    return tok_ll[n_ctx - 1:].sum().item()               # option 片段


@torch.no_grad()
def eval_task(model, tok, name, items, device):
    correct = 0
    for it in items:
        lls = [option_loglikelihood(model, tok, it["ctx"], o, device)
               for o in it["opts"]]
        pred = max(range(len(lls)), key=lambda i: lls[i])
        correct += (pred == it["gold"])
    acc = correct / max(len(items), 1)
    print(f"  {name:16s}: {correct}/{len(items)} = {acc*100:.1f}%")
    return acc


def load_qat_weights(model, ckpt_path):
    from eval_ppl import load_qat_weights as _lw
    _lw(model, ckpt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--split", default="test",
                    help="val(小,调试) / test(正式验收)；mmmlu 用 val/easy/hard")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    langs = args.langs.split(",")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tasks = {}
    tasks.update(build_mmmlu_tasks(langs, args.limit,
                                   "val" if args.split == "test" else args.split))
    tasks.update(build_xnli_tasks(langs, args.limit, args.split))
    tasks.update(build_hellaswag_tasks(langs, args.limit, args.split))
    tasks.update(build_logiqa_tasks(langs, args.limit, args.split))
    if not tasks:
        raise SystemExit("无任务：检查 Qwen--P-MMEval 目录与 --langs")
    print(f"[Eval] 任务: {list(tasks.keys())} (每任务 ≤{args.limit} 题)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=args.device, low_cpu_mem_usage=True,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16)

    tag = "基线"
    results = {}
    for name, items in tasks.items():
        results[f"{name}|base"] = eval_task(model, tok, f"{name} [基线]",
                                            items, args.device)
    if args.ckpt:
        load_qat_weights(model, args.ckpt)
        tag = os.path.basename(args.ckpt)
        for name, items in tasks.items():
            results[f"{name}|quant"] = eval_task(model, tok, f"{name} [量化]",
                                                 items, args.device)

    print("\n===== 汇总 =====")
    for name in tasks:
        b = results.get(f"{name}|base")
        q = results.get(f"{name}|quant")
        if b is not None and q is not None:
            print(f"{name:16s}: {b*100:5.1f}% -> {q*100:5.1f}%  "
                  f"(Δ{(q-b)*100:+.1f}pt)")


if __name__ == "__main__":
    main()
