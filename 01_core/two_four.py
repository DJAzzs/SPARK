"""SPARK v3 — 2:4 结构化稀疏（SPFP2 层）

核心观察：SPFP2 量化后非零率 ~55%，距 2:4 阈值（50%）仅 5 个点。

语义顺序（关键）：**先量化后 2:4 约束**——量化自然产生零（小值落零区），
再对量化值施加"每组至多 2 非零"。组内非零数可为 0/1/2。

v3 位流格式（双块 32 权重 = 56 bit = 7 字节，1.75 bit/权重）：
  模式位: 8 组 × 4 bit
    0-5  : 双非零（C(4,2)=6 种位置组合）
    8-11 : 单非零（4 种位置，符号取 s0）
    15   : 全零
  符号位: 8 组 × 2 bit（非零值的符号）
  指数:   2 块 × 4 bit（沿用 SPFP2 窗口搜索语义）

roundtrip: unpack(pack(w)) == two_four_apply(quantize(w))  bit 级一致。
"""
from __future__ import annotations

import torch

# 双非零模式: idx 0-5 → (pos_a, pos_b)；单非零: idx 8-11 → (pos, None)
_DUAL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_DUAL_LUT = torch.zeros(4, 4, dtype=torch.long)
for _i, (_a, _b) in enumerate(_DUAL):
    _DUAL_LUT[_a][_b] = _i
    _DUAL_LUT[_b][_a] = _i

BYTES_PER_PAIR = 7   # 56 bit / 32 权重


def two_four_apply(q: torch.Tensor) -> torch.Tensor:
    """对已量化（三元）值施加 2:4 约束：每 4 连续权重至多 2 非零。

    全向量化——按 |值| 保留每组 Top-2，其余置零。
    """
    shape = q.shape
    flat = q.reshape(-1, 4)
    # 每组按绝对值排序，保留 top-2
    order = flat.abs().argsort(dim=1, descending=True)
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep.scatter_(1, order[:, :2], True)
    return (flat * keep).reshape(shape)


def two_four_mask_raw(w: torch.Tensor) -> torch.Tensor:
    """连续值版（工程用）：每 4 权重保留 |w| Top-2。"""
    shape = w.shape
    flat = w.reshape(-1, 4)
    order = flat.abs().argsort(dim=1, descending=True)
    keep = order[:, :2]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, keep, True)
    return (flat * mask).reshape(shape)


class TwoFourSTE(torch.autograd.Function):
    """量化 → 2:4 约束的 STE（梯度直通，拓扑可竞争）。"""

    @staticmethod
    def forward(ctx, x):
        from block_fp2_emu import channel_fake_quantize
        return two_four_apply(channel_fake_quantize(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def two_four_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """v3 QAT 前向（可导）。"""
    if x.requires_grad:
        return TwoFourSTE.apply(x)
    from block_fp2_emu import channel_fake_quantize
    return two_four_apply(channel_fake_quantize(x))


def _encode_group(grp_q: torch.Tensor):
    """编码一个 4 元组（三元值）→ (mode4, s0, s1)。"""
    nz = (grp_q != 0).nonzero().flatten()
    if nz.numel() == 0:
        return 15, 0, 0
    if nz.numel() == 1:
        p = nz[0].item()
        return 8 + p, int(grp_q[p].item() < 0), 0
    a, b = nz[0].item(), nz[1].item()
    return int(_DUAL_LUT[a][b]), int(grp_q[a].item() < 0), int(grp_q[b].item() < 0)


def _quantize_block(wblock: torch.Tensor, e: int) -> torch.Tensor:
    """按指数 e 把 16 权重量化为三元值。"""
    from block_fp2_emu import BIAS
    scale = 2.0 ** (e - BIAS)
    return torch.where(wblock.abs() > scale * 0.5,
                       torch.sign(wblock), torch.zeros_like(wblock)) * scale


def _search_exp(wblock: torch.Tensor, center: int) -> int:
    """窗口 ±2 指数搜索——与 channel_fake_quantize 的搜索口径完全一致
    （误差不含 2:4 约束；量化后再施加约束）。这是 roundtrip 一致的前提。"""
    from block_fp2_emu import BIAS
    best_e, best_err = center, None
    for off in (-2, -1, 0, 1, 2):
        e = max(0, min(15, center + off))
        q = _quantize_block(wblock, e)
        err = ((wblock - q) ** 2).sum().item()
        if best_err is None or err < best_err:
            best_err, best_e = err, e
    return best_e


def two_four_pack(w: torch.Tensor) -> torch.Tensor:
    """打包 2D (oc, ic) → uint8 [ceil(nblocks/2)*7]。

    内部先量化（窗口指数搜索）+ 2:4 约束，与 two_four_fake_quant 值一致。
    """
    from block_fp2_emu import BIAS
    w = w.detach().contiguous().float()
    oc, ic = w.shape
    nb_ic = (ic + 15) // 16
    ic_p = nb_ic * 16
    if ic_p != ic:
        full = torch.zeros(oc, ic_p)
        full[:, :ic] = w
        w = full
    wb = w.view(-1, 16)
    nb = wb.shape[0]
    if nb % 2 == 1:
        wb = torch.cat([wb, torch.zeros(1, 16)])
        nb += 1

    maxv = wb.abs().amax(dim=1).clamp_min(1e-38)
    centers = (torch.log2(maxv) + BIAS - 0.75).round().long().clamp(0, 15)

    out = torch.zeros((nb // 2) * BYTES_PER_PAIR, dtype=torch.uint8)
    for b in range(nb):
        e = _search_exp(wb[b], int(centers[b]))
        q = two_four_apply(_quantize_block(wb[b], e))
        v = 0
        slot = b % 2
        base = slot * 28
        for g in range(4):
            mode, s0, s1 = _encode_group(q[g * 4:(g + 1) * 4])
            v |= mode << (base + 4 * g)
            v |= s0 << (base + 16 + 2 * g)
            v |= s1 << (base + 16 + 2 * g + 1)
        v |= (e & 0xF) << (base + 24)
        pi = b // 2
        cur = 0
        for k in range(7):
            cur |= int(out[pi * 7 + k]) << (8 * k)
        cur |= v
        for k in range(7):
            out[pi * 7 + k] = (cur >> (8 * k)) & 0xFF
    return out


def two_four_unpack(packed: torch.Tensor, n_elements: int,
                    dtype=torch.float32) -> torch.Tensor:
    """逆向：uint8 → 近似权重值。"""
    from block_fp2_emu import BIAS
    np_ = packed.numel() // BYTES_PER_PAIR
    pv = packed.view(np_, 7).to(torch.long)
    v = pv[:, 0] | (pv[:, 1] << 8) | (pv[:, 2] << 16) | (pv[:, 3] << 24) \
        | (pv[:, 4] << 32) | (pv[:, 5] << 40) | (pv[:, 6] << 48)

    out = torch.zeros(np_ * 32, dtype=dtype)
    for slot in range(2):
        base = slot * 28
        e = ((v >> (base + 24)) & 0xF).float()
        scale = torch.pow(2.0, e - BIAS)
        for g in range(4):
            mode = (v >> (base + 4 * g)) & 0x7            # 低3bit 先取
            mode_full = (v >> (base + 4 * g)) & 0xF        # 完整4bit
            s0 = ((v >> (base + 16 + 2 * g)) & 1).float()
            s1 = ((v >> (base + 16 + 2 * g + 1)) & 1).float()
            for r in range(np_):
                m = int(mode_full[r])
                vals = torch.zeros(4, dtype=dtype)
                if m < 6:
                    a, b_ = _DUAL[m]
                    vals[a] = torch.where(s0[r] > 0, -scale[r], scale[r])
                    vals[b_] = torch.where(s1[r] > 0, -scale[r], scale[r])
                elif 8 <= m <= 11:
                    a = m - 8
                    vals[a] = torch.where(s0[r] > 0, -scale[r], scale[r])
                # m == 15: 全零
                out[r * 32 + slot * 16 + g * 4: r * 32 + slot * 16 + g * 4 + 4] = vals
    return out[:n_elements].to(dtype)
