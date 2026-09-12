#!/usr/bin/env python3
"""SPARK 全管道验证（旧入口，已被 04_scripts/run_closed_loop.py 取代）。

说明：
  - 使用新版接口（export_spark_model 现接受 base_model_dir）。
  - 逐层 quantize 使用 ChannelFP2Linear 判断。
  - 依赖 GPU 加速（CPU 会较慢）；优先改用 run_closed_loop.py。
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "02_model"), os.path.join(_HERE, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ.get("SPARK_BASE_MODEL", "/home/dja/桌面/Models/Qwen3.5-4B")

print("[1] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="cpu",
    low_cpu_mem_usage=True, torch_dtype=torch.float32)
print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

from quant_linear import apply_channel_fp2_quant, ChannelFP2Linear
from spark_exporter import export_spark_model
from spark_loader import load_spark_model

print("[2] Converting to ChannelFP2Linear...")
model = apply_channel_fp2_quant(model)

print("[3] Quantizing (candidate-search packing)...")
nq = 0
for name, module in model.named_modules():
    if isinstance(module, ChannelFP2Linear):
        module.quantize()
        nq += 1
print(f"   quantized {nq} ChannelFP2 layers")

out_dir = "/tmp/spark_full_pipeline"
import shutil
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

print("[4] Exporting .spark ...")
export_spark_model(model, out_dir, base_model_dir=BASE)

print("[5] Reloading...")
loaded_model, tok = load_spark_model(out_dir, BASE)

prompt = "2+2="
inputs = tok(prompt, return_tensors='pt')
with torch.no_grad():
    output = loaded_model.generate(**inputs, max_new_tokens=8,
                                   pad_token_id=tok.eos_token_id)
print(f"[6] Result: {tok.decode(output[0])}")
print("\n✅ SPARK full pipeline completed!")
