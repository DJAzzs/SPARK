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

# ---- venv 优先：用 venv 的 python -m torch.distributed.run ----
# torch 是 .pth 继承的（.venv/bin/torchrun 入口不存在），必须用 python -m
# 才能带上 venv 的 sys.path（transformers 5.17 / bitsandbytes）。
if [ -x ".venv/bin/python3" ]; then
    PY="$PWD/.venv/bin/python3"
else
    PY="$(command -v python3)"
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

exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model "$MODEL" \
    --data /home/dja/桌面/SPARK/Dataset \
    --outdir "$OUTDIR" \
    --batch_size 16 \
    --accum 8 \
    --steps "$STEPS" \
    --lr 7e-5 \
    --dtype bf16 \
    --log-every 5 \
    --ppl-data data/wikitext2.txt \
    --ddp
