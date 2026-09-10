#!/usr/bin/env python3
"""SPARK 权重导出（.spark 格式）

测算或实际导出：
  - fp2_bytes   = Σ(FP2量化模块参数量) × (40bit/16)/8 ≈ 0.72GB（对~3B模型）
  - total_bytes = fp2_bytes + scale_index表

用法：
    python export_spark_engine.py --model /path/to/model --测算
"""
from __future__ import annotations

import os, sys, json, glob
sys.path.insert(0, '/home/dja/桌面/SPARK')

def estimate_size(model_dir: str) -> dict:
    """估算导出体积（只读config，不加载权重）."""
    cfg_path = f"{model_dir}/config.json"
    if not os.path.exists(cfg_path):
        return {"error": "not found config"}
    
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    tc = cfg.get("text_config", {})
    num_layers = tc.get("num_hidden_layers", 0)
    hidden_size = tc.get("hidden_size", 0)
    intermediate_size = tc.get("intermediate_size", 0)
    
    # 粗略估算参数量（实际以 safetensors.index为准）
    # Qwen3.5-4B: ~2.6B linear params (excl embed/lm_head)
    # For now, use ratio from Qwen3.5-4B: ~70% dense params => 2.6B
    total_est = 4.6e9   # full model
    dense_ratio = 0.72   # FP2able Linear+Embedding
    fp2_params = total_est * dense_ratio
    
    fp2_bytes = fp2_params * (40/128)   # 36bit/16param -> aligned 40bit
    index_bytes = fp2_params / 16         # ~1B per block(16params)
    
    return {
        "fp2_bytes_GB": fp2_bytes / (1024**3),
        "index_bytes_MB": index_bytes / (1024**2),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--测算", action="store_true")
    args = ap.parse_args()
    
    if not os.path.isdir(args.model):
        print(f"[ERROR] {args.model} not a directory"); return 1
    
    res = estimate_size(args.model)
    if "error" in res:
        print(res["error"]); return 2
    
    print(f"\n=== SPARK .spark Engine Size ===")
    print(f"  fp2_quant_bytes: ~{res['fp2_bytes_GB']:.3f} GB  (Params×40/16/8)")
    print(f"  scale_index:      ~{res['index_bytes_MB']:.1f} MB")
    
    if args.测算:
        return 0
    
    # TODO: 实际导出 .spark 格式（预留接口）
    print("\n[INFO] --测算 模式：仅估算体积，未写入文件")


if __name__ == "__main__":
    main()
