#!/bin/bash
# SPARK QAT 训练启动脚本
#
# 用法：
#   双卡 DDP：   ./04_scripts/run_qat.sh
#   单卡：       ./04_scripts/run_qat.sh <data> <outdir> single
#   自定义参数： 直接 python 03_training/trainer.py --help
set -e

DATA_PATTERN=${1:-"/home/dja/桌面/SPARK/Dataset"}
OUTDIR=${2:-"/home/dja/桌面/SPARK/saves/spark-qat"}
MODE=${3:-"ddp"}   # ddp | single

echo "=== SPARK QAT ==="
echo "  data: $DATA_PATTERN"
echo "  outdir: $OUTDIR"
echo "  mode: $MODE"

cd /home/dja/桌面/SPARK

export PYTHONUNBUFFERED=1   # 实时输出（trainer 内也做了行缓冲双保险）

COMMON="--data $DATA_PATTERN --outdir $OUTDIR --batch_size 16 --steps 15000 --lr 1e-4 --no-checkpointing"
if [ "$MODE" = "single" ]; then
    echo ">>> 单卡模式 (cuda:0)"
    exec python3 -u 03_training/trainer.py $COMMON --device cuda:0
else
    echo ">>> DDP 双卡模式 (torchrun --nproc_per_node=2 --ddp)"
    # 本机双卡 NCCL P2P 通道 hang（已实测），强制走共享内存；其他机器可尝试去掉
    export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抗显存碎片（长/短样本混排时 reserved-unallocated 碎片）
    exec torchrun --standalone --nproc_per_node=2 \
         03_training/trainer.py $COMMON --ddp
fi
