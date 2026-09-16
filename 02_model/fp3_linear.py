"""FP3 E1M1 QAT 层 — 三档混合量化的中间档

格式：3-bit E1M1 码（±{0, 0.5, 1.0, 1.5}）+ FP8 块 scale
密度：3 + 8/16 = 3.5 bit/权重（真 3-bit 打包：8 码 × 3bit = 3 字节）
精度：SNR ~13.8 dB（vs SPFP2 ~7.7, NVFP4 ~18.8）

在 SPARK 三档中的角色：替代大部分 attn/GDN 层的 NVFP4，
砍掉体积大头（4.5→3.5 bit）同时保持非线性保护下的可用精度。
"""
from __future__ import annotations

import torch
from torch import nn
from typing import Optional

try:
    from fp3_emu import fp3_fake_quant, fp3_pack, fp3_unpack
    _FP3_OK = True
except ImportError:
    _FP3_OK = False

_WQ_VERSION = [0]


class FP3QATLinear(nn.Module):
    """FP3 E1M1 QAT 层（STE + 缓存，与 NVFP4QATLinear 同构）。"""

    force_fake_quant: bool = False

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        assert _FP3_OK, "fp3_emu 不可用"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,
                                                 device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self._wq_cache: Optional[torch.Tensor] = None
        self._wq_ver: int = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.force_fake_quant:
            if self._wq_cache is None or self._wq_ver != _WQ_VERSION[0]:
                with torch.no_grad():
                    self._wq_cache = fp3_fake_quant(
                        self.weight.detach()
                    ).to(self.weight.dtype)
                self._wq_ver = _WQ_VERSION[0]
            w_eff = self.weight + (self._wq_cache - self.weight).detach()
            return nn.functional.linear(x, w_eff, self.bias)
        return nn.functional.linear(x, self.weight, self.bias)

    def quantize(self):
        """导出为 3-bit 打包（8 码 × 3bit = 3 字节/块）。"""
        codes, scales = fp3_pack(self.weight.data.detach().float())
        self._fp3_codes = codes
        self._fp3_scales = scales
