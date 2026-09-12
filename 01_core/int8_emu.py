"""INT8 (per-output-channel symmetric) — emu + STE

用于混合量化策略中的 lm_head 档位：
  scale_c = amax(row_c) / 127，w ≈ round(w/scale).clamp(±127) × scale
密度 8 bit/权重，round-trip 误差极小（lm_head 安全牌）。
"""
from __future__ import annotations

import torch


def int8_fake_quant(w: torch.Tensor) -> torch.Tensor:
    """2D (oc, ic) 权重的 per-channel 对称 INT8 round-trip。"""
    w = w.detach().contiguous().float()
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(w / scale).clamp_(-127, 127)
    return q * scale


class INT8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return int8_fake_quant(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def int8_fake_quant_ste(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return INT8STE.apply(x)
    return int8_fake_quant(x)


# ---- PTQ 导出（后置量化：训练 BF16，导出时打包 INT8） --------------------------
def ptq_int8_pack(w: torch.Tensor):
    """lm_head 后置 INT8：返回 (int8 权重, fp32 per-channel scale)。

    w: [oc, ic] BF16/fp32 → q: int8 [oc, ic], scale: fp32 [oc]
    解码 w ≈ q.float() * scale.unsqueeze(1)
    """
    wf = w.detach().contiguous().float()
    scale = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    q = torch.round(wf / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    return q, scale


def ptq_int8_unpack(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(1).to(torch.float32)
