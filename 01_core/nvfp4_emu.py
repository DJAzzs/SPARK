"""NVFP4 (E2M1 + per-16-block FP8 E4M3 scale) — PyTorch 参考 emu + STE

对齐 NVIDIA NVFP4 规范（与 torchao mx_formats / TensorRT-LLM 同口径）：
  - 数据：4-bit 浮点 E2M1，码值 {0, 0.5, 1, 1.5, 2, 3, 4, 6} × ±
  - scale：每 16 元素块一个 FP8 E4M3（float8_e4m3fn）标量，
    scale = block_amax / 6（6 为 E2M1 最大码值），解码 w ≈ code × scale
  - 有效密度 ~4.25 bit/权重（4 bit 数据 + 8 bit scale / 16 块）

QAT：NVFP4STE 前向走 round-trip，反向纯直通（与 SPFP2 的 ChannelFP2STE 同构）。
"""
from __future__ import annotations

import torch

ELEMS_PER_BLOCK = 16
E2M1_MAX = 6.0

# E2M1 非负码值（对称）+ 用于最近邻搜索的边界
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# 相邻码值中点（bucketize 边界）：0.25,0.75,1.25,1.75,2.5,3.5,5
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """最近邻映射到 E2M1 码值（round-to-nearest-even 近似：中点取大值）。"""
    ax = x.abs()
    idx = torch.bucketize(ax, _E2M1_BOUNDS.to(x.device))   # [0..7]
    q = _E2M1_POS.to(x.device)[idx]
    return torch.sign(x) * q


def _fp8_e4m3_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 (float8_e4m3fn) round-trip：scale 本身的量化。"""
    return t.to(torch.float8_e4m3fn).to(torch.float32)


def nvfp4_fake_quant(w: torch.Tensor) -> torch.Tensor:
    """2D (oc, ic) 权重的 NVFP4 round-trip（值域仿真，不解码到 4bit 码字）。

    pad ic 到 16 倍数 → 每 16 块 amax/6 得 scale → FP8 量化 scale
    → w/scale 映射 E2M1 → ×scale 还原。
    """
    w = w.detach().contiguous().float()
    oc, ic = w.shape
    nb = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
    ic_p = nb * ELEMS_PER_BLOCK
    if ic_p != ic:
        full = torch.zeros(oc, ic_p, dtype=w.dtype)
        full[:, :ic] = w
        w = full
    wb = w.view(oc * nb, ELEMS_PER_BLOCK)

    amax = wb.abs().amax(dim=1)
    scale = _fp8_e4m3_roundtrip((amax / E2M1_MAX).clamp_min(1e-30))
    q = _quantize_e2m1(wb / scale.unsqueeze(1))
    out = (q * scale.unsqueeze(1)).view(oc, ic_p)
    return out[:, :ic].contiguous()


class NVFP4STE(torch.autograd.Function):
    """NVFP4 fake-quant + 纯直通 STE（与 ChannelFP2STE 同构）。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return nvfp4_fake_quant(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone()


def nvfp4_fake_quant_ste(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return NVFP4STE.apply(x)
    return nvfp4_fake_quant(x)
