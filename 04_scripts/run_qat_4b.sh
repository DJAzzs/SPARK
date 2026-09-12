#!/bin/bash
# SPARK QAT — Qwen3.5-4B (DeepSpeed ZeRO-2, 双卡)
#
# 显存预算（每卡 96GB）：
#   fp32 权重 16GB + ZeRO-2 分片梯度 8GB + 分片优化器 16GB ≈ 40GB + 激活 → 充裕
# 有效 batch = 2卡 × batch 8 × accum 8 = 128
set -e
cd /home/dja/桌面/SPARK

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1   # 本机双卡 NCCL P2P 通道 hang（已实测），走共享内存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抗显存碎片

MODEL=${MODEL:-/home/dja/桌面/Models/Qwen3.5-4B}
OUTDIR=${OUTDIR:-/home/dja/桌面/SPARK/saves/spark-qat-4b}
STEPS=${STEPS:-15000}

echo "=== SPARK QAT 4B (DeepSpeed ZeRO-2) ==="
echo "  model: $MODEL"
echo "  outdir: $OUTDIR"

exec torchrun --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model "$MODEL" \
    --data /home/dja/桌面/SPARK/Dataset \
    --outdir "$OUTDIR" \
    --batch_size 8 \
    --accum 8 \
    --steps "$STEPS" \
    --lr 1e-4 \
    --dtype bf16 \
    --log-every 5 \
    --ppl-data data/wikitext2.txt \
    --deepspeed
