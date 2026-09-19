"""FP3 E1M1 — 中间精度档（3 幅度级，密度 3.5 bit/权重）

格式（对齐 NVFP4 的块结构）：
  数据：3-bit E1M1，码值 ±{0, 0.5, 1.0, 1.5}
  scale：FP8 E4M3 per 16-element block（与 NVFP4 同构）
  解码：value = E1M1_code × scale
  密度：3 + 8/16 = 3.5 bit/权重

码表（无符号 3bit: sign=0 正，sign=1 负，2bit magnitude）:
  000 = +0.0   100 = -0.0
  001 = +0.5   101 = -0.5
  010 = +1.0   110 = -1.0
  011 = +1.5   111 = -1.5
"""
from __future__ import annotations

import torch

ELEMS_PER_BLOCK = 16
_E1M1_MAG = [0.0, 0.5, 1.0, 1.5]      # 2-bit magnitude codes
_BOUNDS = [0.25, 0.75, 1.25]            # nearest-neighbor boundaries


def fp3_fake_quant(w: torch.Tensor) -> torch.Tensor:
    """2D (oc, ic) 权重的 FP3 round-trip（值域仿真）。"""
    w = w.detach().contiguous().float()
    oc, ic = w.shape
    nb = (ic + 15) // 16
    ic_p = nb * 16
    if ic_p != ic:
        full = torch.zeros(oc, ic_p); full[:, :ic] = w; w = full
    wb = w.view(oc * nb, 16)

    amax = wb.abs().amax(dim=1)
    scales = (amax / 1.5).clamp_min(1e-30)   # max code = 1.5
    scales = scales.to(torch.float8_e4m3fn).to(torch.float32).unsqueeze(1)
    xs = wb / scales

    mag = torch.bucketize(xs.abs(), torch.tensor(_BOUNDARY_LIST, device=w.device))
    sign = (xs < 0).long()
    codes = (sign << 2) | mag                    # 3-bit code
    vals = torch.tensor(_FULL_TABLE, device=w.device)[codes]
    return (vals * scales.squeeze(1).unsqueeze(1)).view(oc, ic_p)[:, :ic]


_BOUNDARY_LIST = _BOUNDS
_FULL_TABLE = [0.0, 0.5, 1.0, 1.5, -0.0, -0.5, -1.0, -1.5]


def fp3_pack(w: torch.Tensor):
    """打包为 (codes uint8[nibbles], scales fp8[nblocks])。4bit/weight 存储
    （实际只用 3bit 值，第 4bit 恒 0，便于 nibble 对齐）。"""
    w = w.detach().contiguous().float()
    oc, ic = w.shape
    nb = (ic + 15) // 16
    ic_p = nb * 16
    if ic_p != ic:
        full = torch.zeros(oc, ic_p); full[:, :ic] = w; w = full
    wb = w.view(oc * nb, 16)

    amax = wb.abs().amax(dim=1)
    scales_fp8 = (amax / 1.5).clamp_min(1e-30).to(torch.float8_e4m3fn)
    s = scales_fp8.to(torch.float32).unsqueeze(1)
    xs = wb / s

    mag = torch.bucketize(xs.abs(), torch.tensor(_BOUNDARY_LIST, device=w.device))
    sign = (xs < 0).long()
    codes3 = (sign << 2) | mag                    # 3-bit codes [0..7]
    flat = codes3.reshape(-1).to(torch.uint8)
    # nibble packing: 2 codes per byte
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    codes = (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)
    return codes.contiguous(), scales_fp8.contiguous()


def fp3_unpack(codes: torch.Tensor, scales: torch.Tensor,
               oc: int, ic: int) -> torch.Tensor:
    """逆向。"""
    n = oc * ((ic + 15) // 16) * 16
    lo = (codes & 0x0F).long()
    hi = (codes >> 4).long()
    flat = torch.stack([lo, hi], dim=1).reshape(-1)[:n]
    table = torch.tensor(_FULL_TABLE, device=codes.device)[flat.clamp(max=7)]
    s = scales.to(torch.float32).repeat_interleave(16)
    return (table * s[:n]).view(oc, -1)[:, :ic].contiguous()


class FP3STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fp3_fake_quant(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def fp3_fake_quant_ste(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return FP3STE.apply(x)
    return fp3_fake_quant(x)


# ===== 真 3-bit 打包（8 码 × 3bit = 3 字节，零浪费）=====

def fp3_pack_3bit(w):
    """真 3-bit: 8 码打包为 3 字节。密度 = 3 + 8/16 = 3.5 bit/权重。"""
    w = w.detach().contiguous().float()
    oc, ic = w.shape
    nb = (ic + 15) // 16
    ic_p = nb * 16
    if ic_p != ic:
        full = torch.zeros(oc, ic_p); full[:, :ic] = w; w = full
    wb = w.view(oc * nb, 16)

    amax = wb.abs().amax(dim=1)
    scales_fp8 = (amax / 1.5).clamp_min(1e-30).to(torch.float8_e4m3fn)
    s = scales_fp8.to(torch.float32).unsqueeze(1)
    xs = wb / s

    mag = torch.bucketize(xs.abs(), torch.tensor(_BOUNDARY_LIST, device=w.device))
    sign = (xs < 0).long()
    codes3 = (sign << 2) | mag                    # 3-bit codes [0..7]
    flat = codes3.reshape(-1).to(torch.uint8)

    # 8 码 → 3 字节
    pad = (8 - flat.numel() % 8) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    n_groups = flat.numel() // 8
    fg = flat.view(n_groups, 8).long()
    v = (fg[:, 0] | (fg[:, 1] << 3) | (fg[:, 2] << 6) | (fg[:, 3] << 9) |
         (fg[:, 4] << 12) | (fg[:, 5] << 15) | (fg[:, 6] << 18) | (fg[:, 7] << 21))
    packed = torch.zeros(n_groups * 3, dtype=torch.uint8)
    packed[0::3] = (v & 0xFF).to(torch.uint8)
    packed[1::3] = ((v >> 8) & 0xFF).to(torch.uint8)
    packed[2::3] = ((v >> 16) & 0xFF).to(torch.uint8)
    return packed.contiguous(), scales_fp8.contiguous()


def fp3_unpack_3bit(packed, scales, oc, ic):
    """真 3-bit 解包。"""
    n = oc * ((ic + 15) // 16) * 16
    n_groups = packed.numel() // 3
    pv = packed.view(n_groups, 3).long()
    v = pv[:, 0] | (pv[:, 1] << 8) | (pv[:, 2] << 16)
    codes = torch.zeros(n_groups * 8, dtype=torch.long)
    for i in range(8):
        codes[i::8] = (v >> (3 * i)) & 0x7
    codes = codes[:n]
    table = torch.tensor(_FULL_TABLE, device=packed.device)[codes]
    s = scales.to(torch.float32).repeat_interleave(16)
    return (table * s[:n]).view(oc, -1)[:, :ic].contiguous()
