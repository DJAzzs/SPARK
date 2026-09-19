#!/usr/bin/env python3
"""纯 CPU 打包：ckpt → .spark v2 容器（不加载模型，只处理 state dict）"""
import os, sys, shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_core'))

import torch
from spark_v2_container import save_spark_v2, load_spark_v2, container_size_gb

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'saves/spark-4b-exp1/spark-qat-final.pt'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data/spark-4b-final.spark'
BASE = sys.argv[3] if len(sys.argv) > 3 else '/home/dja/桌面/Models/Qwen3.5-4B'

print(f"[1/3] 加载 ckpt: {CKPT}")
state = torch.load(CKPT, map_location='cpu', weights_only=False)
n_packed = sum(1 for k in state if k.endswith('_packed'))
n_nvfp4 = sum(1 for k in state if k.endswith('_nvfp4_codes'))
n_fp3 = sum(1 for k in state if k.endswith('_fp3_codes'))
n_params = sum(1 for k in state if k.startswith('param::'))
print(f"  SPFP2={n_packed} NVFP4={n_nvfp4} FP3={n_fp3} params={n_params}")

print(f"[2/3] 打包 .spark v2 → {OUT}")
if os.path.exists(OUT):
    shutil.rmtree(OUT)
stats = save_spark_v2(state, OUT, base_model_dir=BASE)
gb = container_size_gb(OUT)
n_w = sum(v.numel() for k, v in state.items()
          if not k.endswith(('_meta', '_scale_index')))
print(f"  raw {stats['raw']/1024**3:.2f}GB → 容器 {gb:.3f}GB "
      f"(≈{gb*8/max(n_w,1):.2f} bit/参数)")

print(f"[3/3] 验证 roundtrip...")
state2 = load_spark_v2(OUT)
ok_p = sum(1 for k in state2 if k.endswith('_packed'))
ok_n = sum(1 for k in state2 if k.endswith('_nvfp4_codes'))
ok_f = sum(1 for k in state2 if k.endswith('_fp3_codes'))
print(f"  回读: SPFP2={ok_p} NVFP4={ok_n} FP3={ok_f}")
assert ok_p == n_packed and ok_n == n_nvfp4 and ok_f == n_fp3, "层数不匹配!"
print(f"  ✅ .spark v2: {OUT} ({gb:.3f}GB)")
