"""SPARK 闭环一致性测试：量化层的打包 ↔ loader 解码 round-trip。

验证目标：
  1. `ChannelFP2Linear.quantize()` 产出的 `_packed_weight` 与 loader 侧解码
     （`decode_layer_weight` / `SparkWeightLoader._decode_weight` 语义）严格一致。
  2. 与单一事实源 `block_fp2_emu`（pack_blockwise_search -> unpack_blockwise）
     的 round-trip 结果 bit 级一致 —— 证明训练侧与推理侧共用同一字节格式。
  3. 覆盖 `ic % 16 != 0`（pad 通道）场景，解码后正确丢弃 pad 还原为 (oc, ic)。
  4. 温和量级权重下，相对 Frobenius 误差在纯三元 block 量化的固有上限内。

运行：cd SPARK && python3 05_tests/test_spark_roundtrip.py
"""
from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from quant_linear import ChannelFP2Linear, FP2Linear, decode_layer_weight
from block_fp2_emu import (BYTES_PER_BLOCK, BYTES_PER_PAIR, ELEMS_PER_BLOCK,
                           pack_blockwise_search, pack_blockwise_paired,
                           unpack_blockwise)


def _make_layer(in_feat, out_feat, bias=True, seed=0, amp=None):
    torch.manual_seed(seed)
    w = (torch.randn(out_feat, in_feat) * 2.0)
    if amp:  # 按通道幅度缩放模拟真实 LLM 层
        for i, m in enumerate(amp[:w.shape[0]]):
            w[i] *= m
    layer = ChannelFP2Linear(in_feat, out_feat, bias=bias)
    with torch.no_grad():
        layer.weight.copy_(w)
    return layer


def _emu_reference(w: torch.Tensor, oc: int, ic: int):
    """统一参考：emu 对 pad 后的权重做候选搜索打包(v2)再解包，还原 (oc, ic)。"""
    from block_fp2_emu import unpack_blockwise_paired
    nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
    ic_pad = nb_ic * ELEMS_PER_BLOCK
    w_full = torch.zeros(oc, ic_pad, dtype=w.dtype)
    w_full[:, :ic] = w
    packed, expv = pack_blockwise_paired(w_full)
    dec = unpack_blockwise_paired(packed, w_full.numel(), dtype=torch.float32)
    return dec.reshape(oc, -1)[:, :ic].reshape_as(w), packed, expv


def _rel_fro_err(w, w_dec):
    err = ((w - w_dec) ** 2).sum().sqrt()
    norm = (w ** 2).sum().sqrt().clamp_min(1e-6)
    return (err / norm).item()


def _tolerance_rel(maxabs):
    """相对容差：较大幅值用宽松相对判据，小幅度退化绝对判据。"""
    return 1e-6 + (maxabs if maxabs > 0 else 1.0) * 2e-4


# ----------------------------------------------------------------------
def test_aligned_matches_emu_exactly():
    """ic % 16 == 0：ChannelFP2Linear 打包 == emu 统一参考（bit 级一致 + 形状）。"""
    oc, ic = 8, 64
    layer = _make_layer(ic, oc, seed=1)
    w = layer.weight.data.detach().float()
    layer.quantize()

    ref, ref_packed, ref_expv = _emu_reference(w, oc, ic)

    # 打包尺寸一致
    assert layer._packed_weight.numel() == ref_packed.numel()
    assert layer._channel_scale_index.numel() == ref_expv.numel()
    # 字节一致（同一格式）——允许浮点选择歧义？指数搜索在相同候选下应完全一致
    assert torch.equal(layer._packed_weight, ref_packed), \
        "packed bytes differ from single-source-of-truth"
    assert torch.equal(layer._channel_scale_index, ref_expv)

    # 解码一致
    w_dec = decode_layer_weight(layer).float()
    assert w_dec.shape == w.shape
    assert torch.allclose(w_dec, ref, atol=_tolerance_rel(w.abs().max())), \
        "decoded differs from emu reference"

    print(f"[aligned ] oc={oc} ic={ic}  packed==emu ✓")


def test_padded_matches_emu_exactly():
    """ic % 16 != 0：pad 情形，打包 == emu 参考，解码丢弃 pad 还原形状。"""
    oc, ic = 5, 20
    layer = _make_layer(ic, oc, seed=7)
    w = layer.weight.data.detach().float()
    layer.quantize()

    ref, ref_packed, ref_expv = _emu_reference(w, oc, ic)

    assert layer._packed_weight.numel() == ref_packed.numel()
    assert torch.equal(layer._packed_weight, ref_packed)
    assert torch.equal(layer._channel_scale_index, ref_expv)

    w_dec = decode_layer_weight(layer).float()
    assert w_dec.shape == w.shape, f"{w_dec.shape} != {w.shape}"
    assert torch.allclose(w_dec, ref, atol=_tolerance_rel(w.abs().max()))
    print(f"[padded  ] oc={oc} ic={ic}  packed==emu ✓  (丢弃 {ic%16} 个 pad)")


def test_fp2_matches_emu():
    """FP2Linear block-wide round-trip 与 emu 一致（v2 paired 格式）。"""
    layer = FP2Linear(48, 12, bias=False)
    torch.manual_seed(3)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(12, 48) * 1.5)
    w = layer.weight.data.detach().float()
    layer.quantize()

    from block_fp2_emu import pack_blockwise_paired, unpack_blockwise_paired
    ref_packed, ref_expv = pack_blockwise_paired(w.view(1, -1))
    assert layer._packed_weight.numel() == ref_packed.numel()
    assert torch.equal(layer._packed_weight, ref_packed)

    w_dec = decode_layer_weight(layer).float()
    assert w_dec.shape == w.shape
    print(f"[fp2     ] oc=12 ic=48  packed==emu ✓")


def test_packed_size_self_consistent():
    """packed 尺寸公式自洽：v2 双块 = ceil(blocks/2)*9 字节，blocks == oc*ceil(ic/16)。"""
    for oc, ic in [(4, 16), (4, 31), (7, 100)]:
        layer = _make_layer(ic, oc, seed=ic)
        layer.quantize()
        expected_blocks = oc * ((ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK)
        expected_v2 = ((expected_blocks + 1) // 2) * BYTES_PER_PAIR
        assert layer._packed_weight.numel() == expected_v2, \
            f"oc={oc} ic={ic}: packed {layer._packed_weight.numel()} != v2 {expected_v2}"
        assert layer._channel_scale_index.numel() == expected_blocks
    print("[size    ] v2 尺寸公式对 ic∈{16,31,100} 自洽 ✓")


def test_moderate_weight_error_bounded():
    """温和量级权重下的绝对 Frobenius 误差在纯三元量化的固有上限内（~线性量化损失）。"""
    # 选量级能被候选指数 0..3 覆盖的权重，避免候选失配导致的"伪精度问题"
    oc, ic = 4, 32
    layer = _make_layer(ic, oc, seed=11, amp=[0.5, 1.0, 1.0, 0.7])
    w = layer.weight.data.detach().float()
    layer.quantize()
    w_dec = decode_layer_weight(layer).float()
    rel = _rel_fro_err(w, w_dec)

    # 三元 block 量化 (±s,0) 相对 Frobenius 误差固有 ~40-60%，这里是温和量级应落在合理带内
    print(f"[error   ] 温和权重 rel_fro_err={rel:.3f} (块量化固有误差)")
    assert rel < 0.90, f"rel err unexpectedly high: {rel:.3f}"


def test_kernel_consistency_if_available():
    """GPU 可用时：emu 解码 == CUDA kernel 解码（一致性收尾）。CPU 下跳过。"""
    if not torch.cuda.is_available():
        print("[kernel  ] 无 CUDA（当前 CPU）——跳过 kernel 一致性检查")
        return
    from build_spark_fp2 import load_kernel
    k = load_kernel()
    oc, ic = 4, 64
    layer = _make_layer(ic, oc, seed=42)
    layer.quantize()
    packed = layer._packed_weight.to("cuda")
    n_total = packed.numel() // BYTES_PER_BLOCK * ELEMS_PER_BLOCK
    k_dec = k.decode_fp32(packed, n_total).cpu().float()
    from quant_linear import _unpack_weight
    e_dec = _unpack_weight(layer._packed_weight, n_total, device="cpu")
    assert torch.allclose(k_dec, e_dec, atol=1e-6), "kernel vs emu decode mismatch"
    print("[kernel  ] emu == CUDA kernel 解码一致 ✓")


def test_export_load_roundtrip(tmp_dir="/tmp/spark_rt_export"):
    """导出 + 加载：loader 解码结果 == 导出前的本地 decode_layer_weight（无数据损坏）。"""
    import shutil
    import torch.nn as nn
    from spark_exporter import export_spark_model
    from spark_loader import SparkWeightLoader

    torch.manual_seed(5)
    oc, ic = 4, 64
    layer = _make_layer(ic, oc, seed=5)
    model = nn.Module()
    setattr(model, "lin", layer)
    layer.quantize()

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    export_spark_model(model, tmp_dir, base_model_dir=None)

    loader = SparkWeightLoader(tmp_dir)
    state = loader.state_dict()
    assert "lin.weight" in state, f"miss key, got {list(state.keys())}"

    w_local = decode_layer_weight(layer).float()
    w_loaded = state["lin.weight"].float()
    assert w_local.shape == w_loaded.shape, f"{w_local.shape} != {w_loaded.shape}"
    assert torch.allclose(w_local, w_loaded, atol=1e-6), "export->load weight corrupt"
    print(f"[export ] export->load 权重一致 ✓ (key='lin.weight')")


if __name__ == "__main__":
    test_aligned_matches_emu_exactly()
    test_padded_matches_emu_exactly()
    test_fp2_matches_emu()
    test_packed_size_self_consistent()
    test_moderate_weight_error_bounded()
    test_kernel_consistency_if_available()
    test_export_load_roundtrip()
    print("\nALL SPARK roundtrip tests passed.")
