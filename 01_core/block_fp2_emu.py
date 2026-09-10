"""SPARK Block-wise Shared-Exponent FP2 — PyTorch 参考实现 (emu)

与 `block_fp2_pack.h` 位布局严格一致（05_tests/test_block_decoding.py 强制 bit-exact）。
作为解码测试的 oracle，也是后续 QAT fake-quant(STE) 的基础。

位布局 (per block, 16 weights shared 1×FP4 exponent):
    bytes[0:4] : uint32 LE = 16 × 2bit mantissa code
    byte [4]   : low nibble = 4-bit exponent (0..15)
    36 bit -> aligned to 40 bit (5 bytes)

mantissa map {00:+1, 01:-1, 10:0, 11:0}
decode value = m * 2^(exp - 2)    （kExpBias == 2）
"""
from __future__ import annotations

import torch
from typing import Tuple

# ---- constants mirroring block_fp2_pack.h ----------------------------------------
ELEMS_PER_BLOCK = 16        # SPARK_ELEMS_PER_BLOCK
MANTISSA_BITS   = 2         # SPARK_MANTISSA_BITS
EXP_BITS        = 4         # SPARK_EXP_BITS
BYTES_PER_BLOCK = 5         # SPARK_BYTES_PER_BLOCK
BIAS            = 2.0       # spark_exp_bias()

# mantissa code table (index by 2-bit code)
_MANT_TABLE: Tuple[float, float, float, float] = (1.0, -1.0, 0.0, 0.0)


def encode_mantissa(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """2bit 尾数编码 {00:+1,01:-1,10:0,11:0}. w/scale 为 fp32/fp16."""
    out = torch.zeros_like(w).to(torch.uint8)
    out[(w > scale * 0.5)] = 0
    out[(w < -scale * 0.5)] = 1
    # else stays 2 (zero code) -- matches kernel's default branch
    return out


def search_exponent(w: torch.Tensor, candidates=None) -> torch.Tensor:
    """对每个 block 四重指数候选，选使 ||W - dequant(W;exp)||_F 最小的 exp.
    返回 shape=[nblocks] int64 tensor (0..15)."""
    n = w.numel()
    nb = n // ELEMS_PER_BLOCK
    if candidates is None:
        candidates = [e for e in range(4)]              # "四重循环" 默认候选 {0,1,2,3}
    wb = w.view(nb, ELEMS_PER_BLOCK)
    best = torch.zeros(nb, dtype=torch.long, device=w.device)
    best_err = None
    for e in candidates:
        scale = torch.scalar_tensor(2.0 ** (e - BIAS), dtype=w.dtype,
                                    device=w.device)
        deq = _dequant_from_mantissa(wb.abs() > scale * 0.5, wb.sign().to(w.dtype),
                                     e)
        err = ((wb - deq) ** 2).sum(dim=-1)             # per-block squared Frobenius
        if best_err is None:
            best_err = err; 
        else:
            better = err < best_err
            best[better] = e
            best_err = torch.where(better, err, best_err)
    return best


def _dequant_from_mantissa(ones_pos: torch.Tensor, ones_neg: torch.Tensor,
                           expv) -> torch.Tensor:
    """按 2bit code 语义重建块值: pos:+1, neg:-1, else 0; × 2^(exp-BIAS)."""
    m = torch.where(ones_pos, torch.tensor(1.0, device=ones_pos.device),
                    torch.zeros((), dtype=torch.float32,
                                device=ones_pos.device)).to(ones_neg.dtype)
    m = torch.where(ones_neg, torch.tensor(-1.0, device=ones_neg.device), m).to(
        ones_pos.dtype)
    scale = (2.0 ** (expv - BIAS))
    return (m * scale)


def pack_blockwise(w: torch.Tensor) -> torch.Tensor:
    """pack 权重张量为字节块。w: contiguous fp32/fp16, numel % 16 == 0.
    返回 uint8 [n_blocks*5] LE packed（尾数+指数）。"""
    w = w.detach().contiguous()
    n = w.numel(); assert n % ELEMS_PER_BLOCK == 0
    nb = n // ELEMS_PER_BLOCK
    wb = w.view(nb, ELEMS_PER_BLOCK)

    # exponent per block: use magnitude-driven search (same as kernel pack)
    maxv = wb.abs().amax(dim=1)                        # [nb]
    l2max = torch.log2(maxv.clamp_min(1e-38))
    exp_raw = (l2max + BIAS).round()
    expv = exp_raw.clamp(0, 15).to(torch.int64)        # [nb]

    scale = (2.0 ** (expv.to(torch.float32) - BIAS))
    codes = encode_mantissa(wb, scale.unsqueeze(-1).expand(nb, ELEMS_PER_BLOCK))   # [nb,16] uint8 {0,1,2}

    # mantissa word: 16×2 bit -> uint32 LE
    shl = torch.arange(0, ELEMS_PER_BLOCK * MANTISSA_BITS, MANTISSA_BITS,
                       device=w.device)                # [0,2,...,30]
    mword = (codes.to(torch.int64).view(nb, ELEMS_PER_BLOCK)
             << shl.view(1, -1)).sum(dim=1)            # [nb] uint32

    out = torch.zeros((nb * BYTES_PER_BLOCK,), dtype=torch.uint8, device=w.device)
    mask256 = (expv & 0x0F).to(torch.int64)
    for bi in range(BYTES_PER_BLOCK):
        if bi < 4:
            v = ((mword >> (8 * bi)) & 0xFF).to(torch.uint8)
        else:
            v = mask256.to(torch.uint8)                 # byte 4: exp only
        out[bi::BYTES_PER_BLOCK] = v
    return out


def unpack_blockwise(packed: torch.Tensor, n_elements: int,
                     dtype=torch.float32) -> torch.Tensor:
    """反向: packed uint8[nb*5] -> fp [n]. 与 CUDA spark_unpack_block bit-exact."""
    nb = packed.numel() // BYTES_PER_BLOCK
    wb = torch.zeros((nb, ELEMS_PER_BLOCK), dtype=dtype,
                     device=packed.device)
    mant_b = torch.zeros(nb, dtype=torch.int64, device=packed.device)
    for bi in range(4):
        col = packed[bi::BYTES_PER_BLOCK].to(torch.int64)  # byte slice
        mant_b |= (col << (8 * bi))
    expv = (packed[4::BYTES_PER_BLOCK] & 0x0F).to(torch.int64)
    scale = (2.0 ** (expv.to(torch.float32) - BIAS)).unsqueeze(-1).to(dtype)

    for i in range(ELEMS_PER_BLOCK):
        code = (mant_b >> (MANTISSA_BITS * i)) & 3
        v = torch.tensor(_MANT_TABLE, device=packed.device)[code].to(dtype)
        wb[:, i] = v * scale[:, 0]
    return wb.reshape(-1)


def decode(w: torch.Tensor) -> torch.Tensor:
    """直接对 fp 张量做 block-FP2 round-trip (pack->unpack), 用于 QAT/误差评估."""
    p = pack_blockwise(w)
    return unpack_blockwise(p, w.numel(), dtype=w.dtype)


# ---- STE fake-quant for future QAT -------------------------------------------------
class BlockFP2STE(torch.autograd.Function):
    """量化感知训练用的 straight-through estimator：
    前向走 block-FP2 round-trip，反向把梯度直通 (忽略量化噪声)。"""
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return decode(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        return (grad_output * (x != 0).to(grad_output.dtype),)


def fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """QAT 前向用的可导 block-FP2。非训练时等价于 decode()."""
    if x.requires_grad:
        return BlockFP2STE.apply(x)
    return decode(x)
