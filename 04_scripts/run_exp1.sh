#!/bin/bash
# SPARK 实验一：三档混合量化 (NVFP4/FP3/SPFP2) 2000 步快速验证
#
# 环境变量（勿删——NCCL_P2P_DISABLE=1 是本机必需的，P2P 通道故障）
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
OUTDIR=${OUTDIR:-saves/spark-4b-exp1}

echo "=== SPARK 实验一: 三档混合 (NVFP4/FP3/SPFP2) ==="
echo "  steps: $STEPS  outdir: $OUTDIR"
echo "  分派: out_proj+embed→NVFP4 | 其余attn/GDN→FP3 | MLP+GDN门控→SPFP2(块8)"

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
    --token-budget 4096 \
    --ppl-data data/wikitext2.txt \
    --ppl-every 500 \
    --log-every 5 \
    --deepspeed \
    --outdir "$OUTDIR"
