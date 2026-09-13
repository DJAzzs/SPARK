#!/bin/bash
# SPARK QAT 续训 — 4B PPL 压缩轮（用户方案 2026-09-13）
#
# 超参（用户设计）:
#   续训步数    +2000 micro steps
#   峰值 LR     3e-5（预热 200 步线性升至峰值，余弦退火到 0）
#   采样        C4:Qwen = 9:1（英文为主，对齐 wikitext 评测分布）
#   损失权重    1:1（归一化均值）
#   C4 数据     全新分片 #2（Dataset-fresh/，与主训 #0/#1 零重叠）
#   Qwen 数据   复用现有 Magpie parquet（软链接）
#   resume      主训 final ckpt（权重全量恢复；AdamW 动量靠 warmup 重建）
set -e
cd /home/dja/桌面/SPARK

# ---- venv 优先：用 venv 的 python -m torch.distributed.run ----
if [ -x ".venv/bin/python3" ]; then
    PY="$PWD/.venv/bin/python3"
else
    PY="$(command -v python3)"
fi

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-saves/spark-qat-4b/spark-qat-final.pt}
OUTDIR=${OUTDIR:-saves/spark-qat-4b-ppl}
STEPS=${STEPS:-2000}

echo "=== SPARK QAT 4B 续训 (PPL 压缩轮) ==="
echo "  resume:  $CKPT"
echo "  data:    Dataset-fresh (C4#2 + Magpie, 9:1)"
echo "  outdir:  $OUTDIR"
echo "  steps:   $STEPS (lr 3e-5 -> 0, warmup 200)"

exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model /home/dja/桌面/Models/Qwen3.5-4B \
    --data /home/dja/桌面/SPARK/Dataset-fresh \
    --sample-c4 9 --sample-qwen 1 \
    --resume-ckpt "$CKPT" \
    --warmup-steps 200 \
    --steps "$STEPS" \
    --lr 3e-5 \
    --min-lr 0 \
    --batch_size 8 \
    --accum 8 \
    --dtype bf16 \
    --quant-mix mixed \
    --quantize-head \
    --token-budget 4096 \
    --ppl-data data/wikitext2.txt \
    --ppl-every 500 \
    --log-every 5 \
    --deepspeed \
    --outdir "$OUTDIR"
