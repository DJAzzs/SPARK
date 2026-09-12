#!/usr/bin/env python3
"""QAT 闭环冒烟测试：STE 训练 → quantize → export(.spark) → load 权重还原.

用一个小型 Llama（2 层）在 CPU 上验证：
  1. QAT 训练层可导、loss 下降（STE 前向/反向）。
  2. 训练后 quantize() 生成 packed。
  3. export_spark_model 导出 .spark。
  4. SparkWeightLoader 解出的权重 与 导出前 decode_layer_weight 一致（无损坏）。
运行：TRITON_CACHE_DIR=/tmp/tc python3 05_tests/test_qat_smoke.py
"""
from __future__ import annotations
import os, sys, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

torch.manual_seed(0)
from transformers import LlamaConfig, LlamaForCausalLM
from quant_linear import (apply_channel_fp2_qat, ChannelFP2QATLinear,
                          decode_layer_weight)
from spark_exporter import export_spark_model
from spark_loader import SparkWeightLoader

cfg = LlamaConfig(hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, vocab_size=1000,
                  max_position_embeddings=256)

# base 原始模型
base_dir = "/tmp/qat_base_dir"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)
LlamaForCausalLM(cfg).save_pretrained(base_dir)
print("[1] base saved")

m = LlamaForCausalLM(cfg)
apply_channel_fp2_qat(m)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
ids = torch.randint(0, 1000, (1, 32)).long()
losses = []
for it in range(2):
    opt.zero_grad()
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    opt.step()
    losses.append(out.loss.item())
print(f"[2] STE QAT train losses: {[round(x,3) for x in losses]}")
assert all(torch.isfinite(torch.tensor(losses)))

# 3) quantize + 记录本地解码基准
with torch.no_grad():
    for _, mod in m.named_modules():
        if isinstance(mod, ChannelFP2QATLinear):
            mod.quantize()
m.eval()

local_dec = {}
for name, mod in m.named_modules():
    if isinstance(mod, ChannelFP2QATLinear):
        local_dec[name.replace('.weight', '') + '.weight'] = \
            decode_layer_weight(mod).float()
print(f"[3] quantized {len(local_dec)} layers")

# 4) export
out_dir = "/tmp/qat_spark_out"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
export_spark_model(m, out_dir, base_model_dir=base_dir)

# 5) loader 解码
loader = SparkWeightLoader(out_dir)
state = loader.state_dict()

missing = [k for k in local_dec if k not in state]
assert not missing, f"loader 缺 key: {missing}"
max_rel = 0.0
for k, w_local in local_dec.items():
    w_loaded = state[k].float()
    rel = ((w_local - w_loaded) ** 2).sum().sqrt() / \
        (w_local ** 2).sum().sqrt().clamp_min(1e-6)
    max_rel = max(max_rel, rel.item())
print(f"[6] loader 解码与导出前最大相对误差: {max_rel:.2e}")
assert max_rel < 1e-4, "QAT 权重导出->加载损坏"
print("\n✅ QAT -> quantize -> export -> loader 权重还原 闭环通过")


if __name__ == "__main__":
    pass
