"""SPARK Block-FP2 / ChannelFP2 Quantized Linear 层"""
from __future__ import annotations

import torch
from torch import nn
from typing import Optional

try:
    from .build_spark_fp2 import load_kernel
    _kernel = load_kernel()
    _KERNEL_AVAILABLE = True
except Exception:
    _KERNEL_AVAILABLE = False


def quantize_tensor_blockwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nb = w.numel() // 16
    wb = w.view(nb, 16)
    
    maxv = wb.abs().amax(dim=1)
    l2max = torch.log2(maxv.clamp_min(1e-38))
    exp_raw = (l2max + 2.0 - 1.0).round()
    expv_block = exp_raw.clamp(0, 15).to(torch.int64)
    
    scale = 2.0 ** (expv_block.to(torch.float32) - 2.0)
    
    codes = torch.zeros_like(wb, dtype=torch.uint8)
    threshold = scale.unsqueeze(-1) * 0.5
    codes[wb > threshold] = 0
    codes[wb < -threshold] = 1
    
    shl = torch.arange(0, 32, 2, device=w.device)
    mword = (codes.to(torch.int64) << shl.view(1, -1)).sum(dim=1)
    
    out = torch.zeros((nb * 5,), dtype=torch.uint8, device=w.device)
    for bi in range(4):
        out[bi::5] = ((mword >> (8 * bi)) & 0xFF).to(torch.uint8)
    out[4::5] = expv_block.to(torch.uint8)
    
    return out, expv_block.to(torch.uint8)


class FP2Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self._packed_weight: Optional[torch.Tensor] = None
        self._scale_index: Optional[torch.Tensor] = None
        
    def quantize(self):
        w = self.weight.data.detach()
        packed, expv = quantize_tensor_blockwise(w)
        
        self._packed_weight = packed
        self._scale_index = expv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _KERNEL_AVAILABLE and self._packed_weight is not None:
            decoded = _kernel.decode_fp32(self._packed_weight.to(x.device),
                                          self.weight.numel())
            w_decoded = decoded.view_as(self.weight)
        else:
            w_decoded = self.weight
        
        return torch.nn.functional.linear(x, w_decoded, self.bias)


class ChannelFP2Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self._packed_weight: Optional[torch.Tensor] = None
        self._channel_scale_index: Optional[torch.Tensor] = None
        
    def quantize(self):
        w = self.weight.data.detach()
        oc, ic = self.out_features, self.in_features
        
        if ic % 16 != 0:
            pad = 16 - (ic % 16)
            w_padded = torch.cat([w, torch.zeros(oc, pad, device=w.device)], dim=1)
        else:
            w_padded = w
        
        nb_ic = (ic + 15) // 16
        wb = w_padded.view(oc * nb_ic, 16)
        
        best_exp = torch.zeros(oc * nb_ic, dtype=torch.uint8, device=w.device)
        for e in range(4):
            scale = 2.0 ** (e - 2.0)
            deq = wb.sign().clamp(min=0) * scale
            deq = torch.where(wb < -scale*0.5, -deq, deq)
            deq = torch.where(wb.abs() <= scale*0.5, 0.0, deq)
            err = ((wb - deq) ** 2).sum(dim=-1)
            if e == 0:
                best_err = err.clone()
                best_exp.copy_(torch.full_like(best_exp, e))
            else:
                better = err < best_err
                best_exp = torch.where(better, torch.full_like(best_exp, e), best_exp)
                best_err = torch.where(better, err, best_err)
        
        shl = torch.arange(0, 32, 2, device=w.device)
        mword = torch.zeros((oc * nb_ic,), dtype=torch.int64, device=w.device)
        codes = torch.zeros_like(wb, dtype=torch.uint8)
        threshold = (2.0 ** (best_exp.to(torch.float32) - 2.0)).unsqueeze(-1) * 0.5
        codes[wb.abs() > threshold] = (wb.sign()[wb.abs() > threshold] + 1).to(torch.uint8) // 2
        
        for i in range(oc * nb_ic):
            mword[i] = (codes[i].to(torch.int64) << shl).sum()
        
        packed = torch.zeros((oc * nb_ic * 5,), dtype=torch.uint8, device=w.device)
        for i in range(oc * nb_ic):
            offset = i * 5
            mant = mword[i].item()
            expv = best_exp[i].item()
            for bi in range(4):
                packed[offset + bi] = (mant >> (8 * bi)) & 0xFF
            packed[offset + 4] = expv
        
        self._packed_weight = packed
        self._channel_scale_index = best_exp.to(torch.uint8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _KERNEL_AVAILABLE and self._packed_weight is not None:
            decoded = _kernel.decode_fp32(self._packed_weight.to(x.device),
                                          self.weight.numel())
            w_decoded = decoded.view_as(self.weight)
        else:
            w_decoded = self.weight
        
        return torch.nn.functional.linear(x, w_decoded, self.bias)


def apply_channel_fp2_quant(model: nn.Module) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if 'lm_head' not in name and 'embed' not in name.lower():
                qlayer = ChannelFP2Linear(
                    module.in_features, module.out_features,
                    module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype)
                with torch.no_grad():
                    qlayer.weight.copy_(module.weight)
                    if module.bias is not None:
                        qlayer.bias.copy_(module.bias)
                setattr(model, name, qlayer)
        else:
            apply_channel_fp2_quant(module)
    return model
