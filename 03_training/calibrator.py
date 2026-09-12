"""SPARK 离线校准器：搜索每Block最优共享指数

对每层的每一个Block，尝试4种指数候选值（0,1,2,3），选取使 ||原始权重 - 解码权重||_F 最小的指数，
结果存入 scale_index_table.pt。此步骤只跑一次，推理时直接查表。

用法：
    python 03_training/calibrator.py --model /home/dja/桌面/Models/Qwen3.5-4B
"""
from __future__ import annotations

import os, sys, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)                 # SPARK/
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "01_core"),
           os.path.join(_PROJECT_ROOT, "02_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch
from quant_linear import ChannelFP2Linear, FP2Linear


@torch.no_grad()
def calibrate_layer(layer: torch.nn.Module, name: str) -> dict:
    """对单层执行指数搜索，返回 scale_index_table entry."""
    if isinstance(layer, (ChannelFP2Linear, FP2Linear)):
        # 已经量化过的层跳过
        return None
    
    state = {}
    
    # 只处理 weight 矩阵（bias 不量化）
    w = layer.weight.data.detach().float()
    oc, ic = w.shape
    
    # 对每一个 output channel，搜索前16 weights的最优指数
    nb_ic = (ic + 15) // 16
    scale_idx = torch.zeros((oc, nb_ic), dtype=torch.uint8, device=w.device)
    
    for ch in range(oc):
        for bi in range(nb_ic):
            start = bi * 16
            end = min(start + 16, ic)
            block = w[ch, start:end].clone()
            
            # 四重循环搜索指数
            best_exp = 0; best_err = float('inf')
            for e in range(16):
                scale = 2.0 ** (e - 2.0)
                deq = torch.where(block > scale*0.5, torch.tensor(scale),
                                 torch.zeros_like(block))
                deq = torch.where(block < -scale*0.5, torch.tensor(-scale), deq)
                err = ((block - deq) ** 2).sum().item()
                if err < best_err:
                    best_err = err; best_exp = e
            
            scale_idx[ch, bi] = best_exp
    
    state['scale_index'] = scale_idx.cpu()
    return {name: state}


@torch.no_grad()
def calibrate_model(model: torch.nn.Module) -> dict:
    """校准全模型，返回 scale_index_table."""
    table = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, ChannelFP2Linear)):
            # 对非 embed/lm_head 走 channel-wise
            if 'embed' not in name.lower() and 'lm_head' not in name:
                entry = calibrate_layer(module, name)
                if entry: table.update(entry)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF checkpoint path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/home/dja/桌面/SPARK/data/scale_index_table.pt")
    args = ap.parse_args()
    
    from transformers import AutoModelForCausalLM
    print(f"[LOAD] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": "cpu"}, torch_dtype=torch.float32,
        low_cpu_mem_usage=True)
    model.eval()
    
    print("[CALIBRATE] searching exponents per block ...")
    table = calibrate_model(model)
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(table, args.out)
    size_mb = os.path.getsize(args.out) / (1024**2)
    print(f"[DONE] saved {len(table)} entries -> {args.out} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
