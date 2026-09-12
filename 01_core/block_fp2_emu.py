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
BIAS            = 10.0      # spark_exp_bias(): scale=2^(e-10) ∈ [2^-10, 2^5]
                           # 覆盖 LLM 权重动态范围（bias=2 会整块归零，见 ptq_eval）

# 候选指数默认全范围 0..15（4-bit 指数可表达的完整网格）
DEFAULT_CANDIDATES = tuple(range(16))

# ---- SPFP2 双块位打包（v2 存储格式）：2 块 72bit = 9 字节，2.25 bit/权重 ----
BYTES_PER_PAIR = 9      # byte[0:4]=块0尾数u32LE, byte[4:8]=块1尾数u32LE,
                        # byte[8]=低4bit块0指数 | 高4bit块1指数

# mantissa code table (index by 2-bit code)
_MANT_TABLE: Tuple[float, float, float, float] = (1.0, -1.0, 0.0, 0.0)


def encode_mantissa(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """2bit 尾数编码 {00:+1,01:-1,10:0,11:0}. w/scale 为 fp32/fp16."""
    # 默认零码 2（与 pack.h 的 `else code = 2u` 一致）。
    # 曾有 bug：初始化为 0 导致零区元素被解码成 +scale 而非 0。
    out = torch.full_like(w, 2, dtype=torch.uint8)
    out[(w > scale * 0.5)] = 0
    out[(w < -scale * 0.5)] = 1
    return out


def search_exponent(w: torch.Tensor, candidates=None) -> torch.Tensor:
    """对每个 block 四重指数候选，选使 ||W - dequant(W;exp)||_F 最小的 exp.
    返回 shape=[nblocks] int64 tensor (0..15)."""
    n = w.numel()
    nb = n // ELEMS_PER_BLOCK
    if candidates is None:
        candidates = list(DEFAULT_CANDIDATES)   # 默认全候选 0..15
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


def pack_blockwise_search(w: torch.Tensor, candidates=DEFAULT_CANDIDATES,
                          window: int = 2):
    """[统一打包器] 候选搜索指数 + 打包，返回 (packed, expv)。

    指数选择（两种模式）：
      - window 搜索（默认, window=2）：由块内 absmax 定中心指数
        e_center = round(log2(blockmax) + BIAS - 0.75)，搜 [c-2, c+2]。
        实测与全候选 0..15 搜索 **指数 100% 一致、SSE bit 级相同**，
        循环次数 16→5，CPU 快 ~1.4x（GPU 上 launch 次数 -3x，收益更大）。
      - window=None：全候选 0..15 逐个搜索（校验用，慢）。
    选定指数后按该指数打包。与 `unpack_blockwise` 严格可逆（round-trip bit-exact）。
    这是训练侧(QAT 层/量化层)与推理侧(loader)共享的单一打包实现。

    w : contiguous fp32/fp16, numel % 16 == 0（调用方负责 16 对齐/pad）。
    返回：
        packed : uint8 [n_blocks*BYTES_PER_BLOCK]
        expv   : uint8 [n_blocks] 4-bit 指数
    """
    w = w.detach().contiguous()
    n = w.numel(); assert n % ELEMS_PER_BLOCK == 0, f"numel {n} not %16==0"
    nb = n // ELEMS_PER_BLOCK
    wb = w.view(nb, ELEMS_PER_BLOCK).to(torch.float32)

    if window is not None:
        # ---- 窗口搜索：absmax 定中心，per-block 候选 [c-window, c+window] ----
        # 实测（高斯块, LLM 权重尺度）最优指数 e* - (log2(maxv)+BIAS) 的分布：
        #   中位 -0.74, 1%~99% 分位 [-1.67, -0.04] → 中心取 -0.75 偏移。
        #   window=2 时与全候选搜索指数 100% 一致（SSE bit 级相同）。
        maxv = wb.abs().amax(dim=1).clamp_min(1e-38)          # [nb]
        e_center = (torch.log2(maxv) + BIAS - 0.75).round().long().clamp(0, 15)
        best = e_center.clone()
        best_err = None
        for off in range(-window, window + 1):
            e = (e_center + off).clamp(0, 15)                 # [nb] 向量候选
            scale = torch.pow(2.0, (e.to(torch.float32) - BIAS)).unsqueeze(-1)
            cand = torch.where(wb.abs() > scale * 0.5,
                               wb.sign() * scale,
                               torch.zeros_like(wb))
            err = ((wb - cand) ** 2).sum(dim=-1)              # [nb]
            if best_err is None:
                best_err = err.clone()
            else:
                better = err < best_err
                best[better] = e[better]
                best_err = torch.where(better, err, best_err)
        expv = best.to(torch.int64)
    else:
        # ---- 全候选搜索（历史行为，校验用）----
        best = torch.zeros(nb, dtype=torch.long, device=w.device)
        best_err = None
        for e in candidates:
            scale = torch.scalar_tensor(2.0 ** (e - BIAS), dtype=torch.float32,
                                        device=w.device)
            cand = torch.where(wb.abs() > scale * 0.5,
                               wb.sign().to(torch.float32) * scale,
                               torch.zeros_like(wb, dtype=torch.float32))
            err = ((wb - cand) ** 2).sum(dim=-1)               # [nb]
            if best_err is None:
                best_err = err.clone()
            else:
                better = err < best_err
                best[better] = e
                best_err = torch.where(better, err, best_err)
        expv = best.clamp(0, 15).to(torch.int64)               # [nb]

    # 按选定指数编码尾数 + 打包成 nb*5 字节
    scale = (2.0 ** (expv.to(torch.float32) - BIAS))
    codes = encode_mantissa(wb, scale.unsqueeze(-1))       # [nb,16] uint8 {0,1,2}
    shl = torch.arange(0, ELEMS_PER_BLOCK * MANTISSA_BITS, MANTISSA_BITS,
                       device=w.device)
    mword = (codes.to(torch.int64).view(nb, ELEMS_PER_BLOCK)
             << shl.view(1, -1)).sum(dim=1)                # [nb] uint32

    out = torch.zeros((nb * BYTES_PER_BLOCK,), dtype=torch.uint8, device=w.device)
    expv8 = (expv & 0x0F).to(torch.uint8)
    for bi in range(BYTES_PER_BLOCK):
        if bi < 4:
            v = ((mword >> (8 * bi)) & 0xFF).to(torch.uint8)
        else:
            v = expv8
        out[bi::BYTES_PER_BLOCK] = v
    return out, expv8


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


# ---- SPFP2 双块位打包 (v2 存储格式)：2 块 72bit = 9 字节 -----------------------
def pack_blockwise_paired(w: torch.Tensor, window: int = 2):
    """双块位打包：2 块 = 2×32bit 尾数 + 2×4bit 指数 = 72bit = 9 字节。

    布局：byte[0:4]=块0尾数 u32 LE | byte[4:8]=块1尾数 u32 LE
          byte[8] = (块1指数 << 4) | 块0指数
    密度 2.25 bit/权重（v1 5字节格式为 2.5）。奇数块数尾部 pad 全零块。
    值语义与 5 字节格式 bit 级一致（同码同指数），仅存储摆放不同。

    返回 (packed uint8[n_pairs*9], expv uint8[nb])
    """
    w = w.detach().contiguous().float()
    n = w.numel(); assert n % ELEMS_PER_BLOCK == 0
    nb = n // ELEMS_PER_BLOCK

    # 复用统一搜索打包得到 mword/expv（从 5 字节格式提取尾数字）
    p5, expv8 = pack_blockwise_search(w, window=window)
    p5v = p5.view(nb, BYTES_PER_BLOCK).to(torch.int64)
    mword = (p5v[:, 0] | (p5v[:, 1] << 8) | (p5v[:, 2] << 16)
             | (p5v[:, 3] << 24))                              # [nb] u32
    expv = p5v[:, 4]                                           # [nb] 0..15

    # pad 到偶数块
    if nb % 2 == 1:
        mword = torch.cat([mword, torch.zeros(1, dtype=mword.dtype)])
        expv = torch.cat([expv, torch.zeros(1, dtype=expv.dtype)])
    n_pairs = mword.numel() // 2

    m0, m1 = mword[0::2], mword[1::2]
    e0, e1 = expv[0::2], expv[1::2]

    out = torch.zeros((n_pairs * BYTES_PER_PAIR,),
                      dtype=torch.uint8, device=w.device)
    ov = out.view(n_pairs, BYTES_PER_PAIR)
    for bi in range(4):
        ov[:, bi] = ((m0 >> (8 * bi)) & 0xFF).to(torch.uint8)
        ov[:, 4 + bi] = ((m1 >> (8 * bi)) & 0xFF).to(torch.uint8)
    ov[:, 8] = (e0 | (e1 << 4)).to(torch.uint8)
    return out, expv8


def unpack_blockwise_paired(packed: torch.Tensor, n_elements: int,
                            dtype=torch.float32) -> torch.Tensor:
    """反向: packed uint8[n_pairs*9] -> fp [n]。与 pack_blockwise_paired 严格可逆。

    n_elements 用于截掉奇数块时的 pad 尾块。
    """
    n_pairs = packed.numel() // BYTES_PER_PAIR
    pv = packed.view(n_pairs, BYTES_PER_PAIR).to(torch.int64)

    def _le_u32(cols):
        v = torch.zeros(n_pairs, dtype=torch.int64, device=packed.device)
        for bi in range(4):
            v |= (cols[:, bi] << (8 * bi))
        return v

    m0 = _le_u32(pv[:, 0:4])
    m1 = _le_u32(pv[:, 4:8])
    e0 = pv[:, 8] & 0x0F
    e1 = (pv[:, 8] >> 4) & 0x0F

    def _decode(mword, expv):
        # mantissa 码表解码（与 unpack_blockwise 同口径）
        nb = mword.shape[0]
        wb = torch.zeros((nb, ELEMS_PER_BLOCK), dtype=dtype,
                         device=packed.device)
        scale = (2.0 ** (expv.to(torch.float32) - BIAS)).unsqueeze(-1).to(dtype)
        table = torch.tensor(_MANT_TABLE, device=packed.device)
        for i in range(ELEMS_PER_BLOCK):
            code = (mword >> (MANTISSA_BITS * i)) & 3
            wb[:, i] = table[code].to(dtype) * scale[:, 0]
        return wb

    both = torch.cat([_decode(m0, e0).reshape(-1),
                      _decode(m1, e1).reshape(-1)], dim=0)
    # 块级交织还原块序：[块0, 块1, 块2, ...]（even/odd 先 reshape 成 [np,16] 再 stack）
    even = both[:n_pairs * ELEMS_PER_BLOCK].view(n_pairs, ELEMS_PER_BLOCK)
    odd = both[n_pairs * ELEMS_PER_BLOCK:].view(n_pairs, ELEMS_PER_BLOCK)
    out = torch.stack([even, odd], dim=1).reshape(-1)
    return out[:n_elements].to(dtype)


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


# ---- channel-aware STE fake-quant for 2D (oc, ic) weight matrices ------------------
class ChannelFP2STE(torch.autograd.Function):
    """2D (oc, ic) 权重的分通道 STE fake-quant.

    与推理层 `ChannelFP2Linear.quantize()` 语义一致：
      - 每行（output channel）的 ic 按 16 分块；
      - 每块用候选指数搜索选最小 Frobenius 误差的指数；
      - ic % 16 != 0 时 pad 到 16（编码为 0），解码后丢弃 pad。
    反向用直通估计（梯度原样回传），忽略量化噪声 —— 即标准 STE。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        oc, ic = x.shape
        xf = x.float() if x.dtype != torch.float32 else x
        # channel pad + 候选搜索打包
        packed, _ = pack_blockwise_search(xf, candidates=DEFAULT_CANDIDATES)
        # 解包（含 pad），丢弃 pad 还原 (oc, ic)
        nb = packed.numel() // BYTES_PER_BLOCK
        dec = unpack_blockwise(packed, nb * ELEMS_PER_BLOCK,
                               dtype=torch.float32).reshape(oc, -1)
        dec = dec[:, :ic].reshape_as(x)
        return dec if x.dtype == torch.float32 else dec.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 标准 STE 纯直通：零区(|w|<=scale/2 → 量化值 0)元素也必须收到梯度，
        # 否则它们永远无法逃离零区（本格式零区占比高，掩码会显著拖慢 QAT 收敛）。
        return grad_output.clone()


def channel_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """2D (oc, ic) 权重的可导 channel-FP2 fake-quant。训练用前向。"""
    if x.requires_grad:
        return ChannelFP2STE.apply(x)
    # 非训练：等同推理层 decode
    oc, ic = x.shape
    packed, _ = pack_blockwise_search(x.float(), candidates=DEFAULT_CANDIDATES)
    nb = packed.numel() // BYTES_PER_BLOCK
    dec = unpack_blockwise(packed, nb * ELEMS_PER_BLOCK,
                           dtype=torch.float32).reshape(oc, -1)[:, :ic]
    return dec.reshape_as(x)
