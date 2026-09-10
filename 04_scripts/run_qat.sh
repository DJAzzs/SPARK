#!/bin/bash
set -e

DATA_PATTERN=${1:-"/home/dja/桌面/SPARK/Dataset"}
OUTDIR=${2:-"/home/dja/桌面/SPARK/saves/spark-qat"}

echo "=== SPARK QAT ==="
echo "  data: $DATA_PATTERN"
echo "  outdir: $OUTDIR"

cd /home/dja/桌面/SPARK

exec torchrun --nproc_per_node=2 \
    03_training/trainer.py \
    --data "$DATA_PATTERN" \
    --outdir "$OUTDIR" \
    --batch_size 4 \
    --steps 15000
