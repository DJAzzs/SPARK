#!/bin/bash
# SPARK 实验二（激进版）: SPFP2 主导的三档混合量化
#
# 分派:
#   SPFP2 (2.5bit): MLP + GDN门控 + GDN主投影(qkvz) + attn q/k/v  = ~170 层 71%
#   FP3   (3.5bit): attn o_proj (仅留最少)                          = ~8 层  1%
#   NVFP4 (4.5bit): GDN out_proj + embed                            = ~25 层 17%
#   体积预期: ~1.25 GB
#
# 环境变量（勿删——NCCL_P2P_DISABLE=1 是本机必需的）
set -e
cd /home/dja/桌面/SPARK

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -x ".venv/bin/python3" ]; then
    PY="$PWD/.venv/bin/python3"
else
    PY="$(command -v python3)"
fi

STEPS=${STEPS:-8000}
OUTDIR=${OUTDIR:-saves/spark-4b-exp2}

echo "=== SPARK 实验二: 激进版 (SPFP2 主导, ~1.25GB) ==="
echo "  steps: $STEPS  outdir: $OUTDIR"
echo "  分派: MLP+GDN门控+GDN主投影+attn_qkv→SPFP2 | attn_o→FP3 | out_proj+embed→NVFP4"

exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model /home/dja/桌面/Models/Qwen3.5-4B \
    --data /home/dja/桌面/SPARK/Dataset \
    --steps "$STEPS" \
    --lr 7e-5 \
    --batch_size 8 \
    --accum 8 \
    --dtype bf16 \
    --quant-mix mixed \
    --quantize-head \
    --fp3-tier \
    --fp3-aggressive \
    --token-budget 4096 \
    --ppl-data data/wikitext2.txt \
    --ppl-every 500 \
    --log-every 5 \
    --deepspeed \
    --outdir "$OUTDIR"
