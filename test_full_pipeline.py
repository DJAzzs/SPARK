#!/usr/bin/env python3
import os, sys
sys.path.insert(0, '/home/dja/桌面/远苍')
sys.path.insert(0, '/home/dja/桌面/SPARK')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[1] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/dja/桌面/Models/Qwen3.5-4B", device_map="cpu",
    low_cpu_mem_usage=True, torch_dtype=torch.float32)
print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

from quant_linear import apply_channel_fp2_quant
print("[2] Converting to ChannelFP2Linear...")
model = apply_channel_fp2_quant(model)

print("[3] Calibrating...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and hasattr(module, 'quantize'):
        try: module.quantize()
        except: pass

from spark_exporter import export_spark_model
out_dir = "/home/dja/桌面/SPARK/data/spark_checkpoint"
export_spark_model(model, out_dir)

from spark_loader import load_spark_model
print("[5] Reloading...")
loaded_model, tok = load_spark_model(out_dir, "/home/dja/桌面/Models/Qwen3.5-4B")

prompt = "2+2="
inputs = tok(prompt, return_tensors='pt')
with torch.no_grad():
    output = loaded_model.generate(**inputs, max_new_tokens=8)
print(f"[6] Result: {tok.decode(output[0])}")
print("\n✅ SPARK full pipeline completed!")
