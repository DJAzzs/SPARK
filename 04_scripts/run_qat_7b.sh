#!/bin/bash
# SPARK QAT — Qwen2.5-7B-Instruct (DeepSpeed ZeRO-2, 双卡)
#
# 显存预算（每卡 96GB）：
#   DDP 全量 fp32 需要 ~122GB/卡（装不下）→ ZeRO-2 分片后：
#   fp32 权重 30.4GB + 分片梯度 15.2GB + 分片优化器 30.4GB ≈ 76GB + 激活
#   → batch 2 起步；若 OOM 降到 batch 1 / accum 32
# 有效 batch = 2卡 × batch 2 × accum 16 = 64
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
export NCCL_P2P_DISABLE=1   # 本机双卡 NCCL P2P 通道 hang（已实测），走共享内存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抗显存碎片

MODEL=${MODEL:-/home/dja/桌面/Models/Qwen2.5-7B-Instruct}
OUTDIR=${OUTDIR:-/home/dja/桌面/SPARK/saves/spark-qat-7b}
STEPS=${STEPS:-15000}

echo "=== SPARK QAT 7B (DeepSpeed ZeRO-2) ==="
echo "  model: $MODEL"
echo "  outdir: $OUTDIR"

exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
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
    --deepspeed
