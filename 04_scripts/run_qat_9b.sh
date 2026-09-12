#!/bin/bash
# SPARK QAT — Qwen3.5-9B (DeepSpeed ZeRO-2 + 8-bit AdamW, 双卡)
#
# 显存预算（每卡 96GB）—— 9B 是这套硬件的贴边规模：
#   ZeRO-2 + fp32 AdamW: 36(w) + 9(g bf16/2) + 36(opt/2) ≈ 81GB + 激活 → 贴边危险
#   ZeRO-2 + AdamW8bit : 36(w) + 9(g bf16/2) + 18(opt8bit/2) ≈ 63GB + 激活 → 稳
#   → 默认启用 --optim-8bit（需 bitsandbytes；未安装会自动退回并警告）
# gradient checkpointing 保持开启（激活显存必须省）。
# 有效 batch = 2卡 × batch 2 × accum 16 = 64
# 预计吞吐 ~350 tok/s；STEPS 默认 10000（大模型恢复快于小模型，无需 15000）
set -e
cd /home/dja/桌面/SPARK

# ---- venv 优先（存在则用 .venv 的 torchrun/python）----
if [ -x ".venv/bin/torchrun" ]; then
    TORCHRUN="$PWD/.venv/bin/torchrun"
else
    TORCHRUN="$(command -v torchrun)"
fi


export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1   # 本机双卡 NCCL P2P 通道 hang（已实测），走共享内存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抗显存碎片

MODEL=${MODEL:-/home/dja/桌面/Models/Qwen3.5-9B}
OUTDIR=${OUTDIR:-/home/dja/桌面/SPARK/saves/spark-qat-9b}
STEPS=${STEPS:-10000}

echo "=== SPARK QAT 9B (DeepSpeed ZeRO-2 + AdamW8bit) ==="
echo "  model: $MODEL"
echo "  outdir: $OUTDIR"
echo "  steps: $STEPS"

exec "$TORCHRUN" --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model "$MODEL" \
    --data /home/dja/桌面/SPARK/Dataset \
    --outdir "$OUTDIR" \
    --batch_size 2 \
    --accum 16 \
    --steps "$STEPS" \
    --lr 1e-4 \
    --dtype bf16 \
    --log-every 5 \
    --ppl-data data/wikitext2.txt \
    --optim-8bit \
    --deepspeed
