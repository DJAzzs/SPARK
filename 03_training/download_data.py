"""SPARK 训练数据下载（ModelScope + 本地 prepared 数据支持）

轻量设计：1B tokens 足够 QAT 收敛
"""
from __future__ import annotations

import os, sys, argparse

sys.path.insert(0, '/home/dja/桌面/远苍')
from prepare_data import resolve_data_files as _resolve


def get_qwen_math_small() -> str:
    """从远仓 Dataset 复用少量高质量数学语料 (~1B tokens)
    
    使用 MathCoT + MathInstruct，估算 ~50k samples × 2048 ≈ 1B tokens
    """
    base = "/home/dja/桌面/远苍/Dataset"
    paths = [
        f"{base}/Math-CoT-20k_deepseek_r1_response.parquet",
        f"{base}/MathInstruct.jsonl",
        f"{base}/math.jsonl",
    ]
    files = [p for p in paths if os.path.exists(p)]
    if not files:
        print("注意: 未找到数学语料，建议先下载数据");
        return ""
    
    # 合并为单一 input pattern
    patterns = ','.join(files)
    print(f"[TRAIN_DATA] 使用语料源: {patterns}")
    return patterns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="手动指定数据路径(glob)")
    args = ap.parse_args()
    
    pattern = args.data or get_qwen_math_small()
    if pattern:
        print(f"\n运行训练:\n  torchrun --nproc_per_node=2 04_scripts/run_qat.sh --data \"{pattern}\"")
