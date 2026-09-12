"""SPARK LLaMA-style 包装器，兼容 Qwen2/Qwen3 系列"""
from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


def replace_with_spark_modules(model: nn.Module):
    """递归替换模型中的 Linear -> ChannelFP2Linear。
    
    适配 Qwen2/Qwen3 架构差异：
    - Qwen2: standard MLP (up_proj, down_proj, gate_proj)
    - Qwen3: hybrid linear_attention/full_attention + gate_mlp
    """
    from quant_linear import ChannelFP2Linear
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 跳过 embed/lm_head 的 FP2（可选：也做 embedding quant）
            if 'embed' not in name.lower() and 'lm_head' not in name:
                new_mod = ChannelFP2Linear(
                    module.in_features, module.out_features,
                    module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype)
                with torch.no_grad():
                    new_mod.weight.copy_(module.weight)
                    if module.bias is not None:
                        new_mod.bias.copy_(module.bias)
                setattr(model, name, new_mod)
        else:
            replace_with_spark_modules(module)
    
    return model


class SparkLlamaWrapper(nn.Module):
    """包裹任意 HF LLaMA-family 模型，应用 SPARK量化。"""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        replace_with_spark_modules(self.model)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_for_inference(model_path: str, device: str = "cuda"):
    """加载量化模型用于推理（自动适配 Qwen2/Qwen3）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[SPARK] loading model from {model_path}")
    
    # 根据 model_type 选择适合的加载参数
    config = nn.Module()
    try:
        import json
        with open(f"{model_path}/config.json") as f:
            cfg = json.load(f)
        config.model_type = cfg.get("model_type", "llama")
        print(f"[SPARK] detected model_type: {config.model_type}")
    except Exception as e:
        print(f"[WARN] config load failed: {e}")
        config.model_type = "llama"
    
    # qwen2 使用 device_map auto，qwen3 需要 manual
    if config.model_type == "qwen2":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16,
            trust_remote_code=True)
    else:
        # qwen3 or others
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, low_cpu_mem_usage=True,
            torch_dtype=torch.float16, trust_remote_code=True)
    
    # 应用 SPARK量化 (在模型加载到 device 后)
    if hasattr(model, 'model'):  # Qwen2/Qwen3 通用
        replace_with_spark_modules(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "/home/dja/桌面/Models/Qwen2.5-7B-Instruct", device_map="cpu",
        local_files_only=True, torch_dtype=torch.float32)
    
    print(f"Qwen2.5 loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    
    wrapped = SparkLlamaWrapper(model)
    print("SPARK wrapper applied to Qwen2.5-7B")
