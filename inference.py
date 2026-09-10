"""SPRK FP2 量化推理（不训练，直接加载量化权重）"""
from __future__ import annotations

import os, sys, argparse, time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def quantize_weights(module: torch.nn.Module):
    """递归量化所有Linear层为 ChannelFP2"""
    if '/home/dja/桌面/SPARK' not in sys.path:
        sys.path.insert(0, '/home/dja/桌面/SPARK')
    
    from block_fp2_emu import pack_blockwise, unpack_blockwise
    
    if isinstance(module, torch.nn.Linear):
        w = module.weight.data.detach().float()
        
        packed = pack_blockwise(w.flatten())
        decoded = unpack_blockwise(packed, w.numel(), dtype=w.dtype).view_as(w)
        
        with torch.no_grad():
            module.weight.copy_(decoded.to(module.weight.dtype))
    
    for child in module.children():
        quantize_weights(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/dja/桌面/Models/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="1+2=")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    
    print(f"[SPARK] loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=args.device, low_cpu_mem_usage=True,
        torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("[SPARK] quantizing weights to FP2...")
    start_quant = time.time()
    quantize_weights(model)
    print(f"[SPARK] quantization done in {time.time()-start_quant:.1f}s")
    
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
        
        start_gen = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_time = time.time() - start_gen
        
        total_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        tps = total_tokens / gen_time if gen_time > 0 else 0
        
    result_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\n=== Response ({tps:.1f} tok/s) ===")
    print(result_text)


if __name__ == "__main__":
    main()
