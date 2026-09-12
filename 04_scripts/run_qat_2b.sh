#!/bin/bash
# SPARK QAT — MiniCPM5-2B (DDP 双卡 + gradient checkpointing)
# 2B 必须开 ckpt：no-ckpt 时 batch16×1059tok 激活≈40GB+，总 91GB 爆（实测）；
# 开 ckpt 激活~4GB → 总 ~60GB 稳（代价 ~25% 速度）
#
# 显存预算（每卡 96GB）：
#   2.52B × 16 B/param (fp32 w+g+AdamW) ≈ 40GB 静态 + 激活(无ckpt, batch16) ~10GB
#   → ~55GB，宽裕。无需 DeepSpeed。
# 有效 batch = 2卡 × batch 16 × accum 8 = 256（与 0.5B 基线对齐）
# 步数: 10000（2.5B 恢复快；cosine 已按 STEPS 自动缩放衰减期）
# 模型家族：MiniCPM（验证 SPARK QAT 跨架构泛化）
set -e
cd /home/dja/桌面/SPARK

# ---- venv 优先（存在则用 .venv 的 torchrun/python）----
if [ -x ".venv/bin/torchrun" ]; then
    TORCHRUN="$PWD/.venv/bin/torchrun"
else
    TORCHRUN="$(command -v torchrun)"
fi


export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-/home/dja/桌面/Models/MiniCPM5-2B}
OUTDIR=${OUTDIR:-/home/dja/桌面/SPARK/saves/spark-qat-2b}
STEPS=${STEPS:-10000}

echo "=== SPARK QAT MiniCPM5-2B (DDP) ==="
echo "  model: $MODEL"
echo "  outdir: $OUTDIR"
echo "  steps: $STEPS"

exec "$TORCHRUN" --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model "$MODEL" \
    --data /home/dja/桌面/SPARK/Dataset \
    --outdir "$OUTDIR" \
    --batch_size 16 \
    --accum 8 \
    --steps "$STEPS" \
    --lr 1e-4 \
    --dtype bf16 \
    --log-every 5 \
    --ppl-data data/wikitext2.txt \
    --ddp
