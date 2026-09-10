"""SPARK 权重加载器"""
from __future__ import annotations

import os, sys, torch
sys.path.insert(0, '/home/dja/桌面/远苍')
sys.path.insert(0, '/home/dja/桌面/SPARK')


class SparkWeightLoader:
    """从 .spark 文件还原量化权重。"""
    def __init__(self, spark_dir: str):
        self.spark_dir = spark_dir
        pth = os.path.join(spark_dir, "state_dict.pt")
        assert os.path.exists(pth), f"not found: {pth}"
        self.data = torch.load(pth, map_location="cpu", weights_only=False)
        print(f"[SPARK-LOADER] loaded {len(self.data)} tensors from .spark")
    
    def state_dict(self):
        """返回解码后的 state_dict（FP16/BF16 用于推理）。"""
        state = {}
        for name, packed in self.data.items():
            if name.endswith('_packed'):
                layer_name = name.replace('_packed', '')
                decoded = self._decode_weight(packed)
                state[layer_name] = decoded
        return state
    
    def _decode_weight(self, packed: torch.Tensor) -> torch.Tensor:
        """从 packed uint8 解码为 fp16 weight tensor."""
        import sys as _sys
        _sys.path.insert(0, '/home/dja/桌面/SPARK/01_core')
        try:
            from build_spark_fp2 import load_kernel
            k = load_kernel()
            decoded = k.decode_fp32(packed.to('cuda'), packed.numel()*4)
            return decoded.view(-1).cpu().to(torch.float16)
        except Exception as e:
            print(f"[WARN] CUDA decode failed, use emu: {e}")
            from block_fp2_emu import unpack_blockwise
            nb = packed.numel() // 5
            weights = unpack_blockwise(packed, nb*16, dtype=torch.float32)
            return weights.view(-1).cpu().to(torch.float16)


def load_spark_model(spark_dir: str, hf_base: str):
    """从 .spark 目录加载 FP2 量化模型进行推理。"""
    import sys as _sys
    _sys.path.insert(0, '/home/dja/桌面/SPARK')
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[LOAD] HF base: {hf_base}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_base, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    
    loader = SparkWeightLoader(spark_dir)
    decoded_state = loader.state_dict()
    
    model_state = model.state_dict()
    matched = 0
    for name, weight in decoded_state.items():
        if name in model_state and model_state[name].shape == weight.shape:
            with torch.no_grad():
                model_state[name].copy_(weight.to(model_state[name].dtype))
            matched += 1
    
    print(f"[LOAD] matched {matched}/{len(model_state)} weights")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model.eval(), tokenizer


if __name__ == '__main__':
    loader = SparkWeightLoader("/home/dja/桌面/SPARK/data/spark_checkpoint")
    sd = loader.state_dict()
    print("state dict keys:", list(sd.keys())[:5])
