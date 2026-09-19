"""SPARK Block-FP2 / ChannelFP2 Quantized Linear 层"""
from __future__ import annotations

import os
import sys
import torch
from torch import nn
from typing import Optional

# ---------------------------------------------------------------- paths
# 把项目根与 01_core 加入 sys.path，保证 CUDA kernel 打包器与 emu 可被正确定位。
# 不用相对导入（01_core 不在 02_model 包内），这是修复 "kernel 从未加载" 的关键。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)                    # SPARK/
_CORE_DIR = os.path.join(_PROJECT_ROOT, "01_core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from block_fp2_emu import (                          # noqa: E402
    pack_blockwise_search,
    pack_blockwise_paired,
    unpack_blockwise,
    unpack_blockwise_paired,
    channel_fake_quantize,
    ELEMS_PER_BLOCK,
    BYTES_PER_BLOCK,
    BYTES_PER_PAIR,
)
try:
    from nvfp4_emu import nvfp4_fake_quant          # noqa: E402
    _NVFP4_AVAILABLE = True
except Exception:
    _NVFP4_AVAILABLE = False

try:
    from int8_emu import int8_fake_quant            # noqa: E402
    _INT8_AVAILABLE = True
except Exception:
    _INT8_AVAILABLE = False

try:
    from two_four import two_four_pack, two_four_unpack, two_four_fake_quant  # noqa: E402
    _V3_AVAILABLE = True
except Exception:
    _V3_AVAILABLE = False

try:
    from fp3_linear import FP3QATLinear  # noqa: E402
    _FP3_TIER_OK = True
except Exception:
    _FP3_TIER_OK = False

try:
    from build_spark_fp2 import load_kernel          # noqa: E402
    _kernel = load_kernel()
    _KERNEL_AVAILABLE = True                          # GPU 可用时才为 True
except Exception as _e:
    _kernel = None
    _KERNEL_AVAILABLE = False
    if os.environ.get("SPARK_VERBOSE"):
        print(f"[SPARK] CUDA kernel not loaded, fall back to emu: {_e}")


def quantize_tensor_blockwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-FP2 打包：对每 16 元素 block 用候选搜索选共享指数再打包。

    返回 (packed, expv)：
      packed : uint8 [n_blocks*BYTES_PER_BLOCK]（与 block_fp2_pack.h 位布局严格一致）
      expv   : uint8 [n_blocks]
    """
    w = w.detach().contiguous().float()
    return pack_blockwise_paired(w)


def _detect_format(packed_numel: int, n_elements: int) -> str:
    """按尺寸无歧义检测打包格式（仅 nb=0 时两式相等，实际不发生）。"""
    nb = (n_elements + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
    if packed_numel == nb * BYTES_PER_BLOCK:
        return "v1"                       # 5 字节/块 (2.5 bit/权重)
    if packed_numel == ((nb + 1) // 2) * BYTES_PER_PAIR:
        return "v2"                       # 9 字节/双块 (2.25 bit/权重)
    raise ValueError(
        f"packed 尺寸 {packed_numel} 与 n_elements {n_elements} 不匹配 "
        f"(v1 期望 {nb*BYTES_PER_BLOCK}B, v2 期望 {((nb+1)//2)*BYTES_PER_PAIR}B)")


def _unpack_weight(packed: torch.Tensor, n_elements: int,
                   device: str | torch.device = "cpu") -> torch.Tensor:
    """把 packed 字节解码回权重（自动检测 v1/v2 格式）。

    v1 (5字节/块) 且 CUDA kernel 可用时走 kernel decode；
    v2 (9字节/双块) 目前走 emu（kernel 待同步支持）。
    """
    fmt = _detect_format(packed.numel(), n_elements)
    if fmt == "v1":
        use_kernel = (_KERNEL_AVAILABLE and _kernel is not None
                      and torch.cuda.is_available())
        if use_kernel:
            try:
                decoded = _kernel.decode_fp32(packed.to(device), n_elements)
                return decoded.cpu().float()
            except Exception:
                pass
        return unpack_blockwise(packed.cpu(), n_elements, dtype=torch.float32)
    return unpack_blockwise_paired(packed.cpu(), n_elements,
                                   dtype=torch.float32)


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
        if self._packed_weight is not None and self._use_kernel_for(x):
            decoded = _kernel.decode_fp32(self._packed_weight.to(x.device),
                                          self.weight.numel()) if _detect_format(
                self._packed_weight.numel(),
                ((self.weight.numel() + ELEMS_PER_BLOCK - 1)
                 // ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK) == "v1" else \
                _unpack_weight(self._packed_weight,
                               ((self.weight.numel() + ELEMS_PER_BLOCK - 1)
                                // ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK,
                               device=x.device).to(x.device)
            w_decoded = decoded.view_as(self.weight)
        else:
            w_decoded = self.weight

        return torch.nn.functional.linear(x, w_decoded, self.bias)

    def _use_kernel_for(self, x: torch.Tensor) -> bool:
        """是否用 CUDA kernel 解码（需 kernel 可用 + x 在 CUDA 上）。"""
        return (
            _KERNEL_AVAILABLE and _kernel is not None
            and torch.cuda.is_available() and x.is_cuda
        )


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
        """Channel-FP2 打包，与 emu/unpack 严格一致。

        布局约定：每行（output channel）的 in_features 按 16 长度分块。
        若 ic % 16 != 0，末尾 pad 到 16（编码为 0），解码后丢弃多出的 pad 元素。
        packed 尺寸 = (oc * ceil(ic/16)) * BYTES_PER_BLOCK，解码需按同样块序展平。
        """
        w = self.weight.data.detach().float()
        oc, ic = self.out_features, self.in_features

        nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
        ic_padded = nb_ic * ELEMS_PER_BLOCK
        if ic_padded != ic:
            w_full = torch.zeros(oc, ic_padded, device=w.device, dtype=w.dtype)
            w_full[:, :ic] = w
        else:
            w_full = w

        # 按 [oc, ic_block] 展平，逐 16 块候选搜索打包（v2 双块位打包 2.25bit）
        packed, expv = pack_blockwise_paired(w_full)

        self._packed_weight = packed
        self._channel_scale_index = expv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed_weight is not None and self._use_kernel_for(x):
            decoded = _kernel.decode_fp32(self._packed_weight.to(x.device),
                                          self.weight.numel()) if _detect_format(
                self._packed_weight.numel(),
                ((self.weight.numel() + ELEMS_PER_BLOCK - 1)
                 // ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK) == "v1" else \
                _unpack_weight(self._packed_weight,
                               ((self.weight.numel() + ELEMS_PER_BLOCK - 1)
                                // ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK,
                               device=x.device).to(x.device)
            w_decoded = decoded.view_as(self.weight)
        else:
            w_decoded = self.weight

        return torch.nn.functional.linear(x, w_decoded, self.bias)

    def _use_kernel_for(self, x: torch.Tensor) -> bool:
        """是否用 CUDA kernel 解码（需 kernel 可用 + x 在 CUDA 上）。"""
        return (
            _KERNEL_AVAILABLE and _kernel is not None
            and torch.cuda.is_available() and x.is_cuda
        )


# ---- QAT fake-quant 缓存版本机制 -------------------------------------------
# 梯度累积组内（两次 optimizer.step() 之间）权重不变，fake-quant 结果可复用。
# trainer 每次 optimizer.step() / engine.step() 后调用 bump_wq_version()，
# 各 QAT 层在下一次 forward 时才重算量化权重。累积 8 步时省 ~8x fake-quant 计算。
_WQ_VERSION = [0]

def bump_wq_version():
    """optimizer.step() 之后调用：使所有 QAT 层的 fake-quant 缓存失效。"""
    _WQ_VERSION[0] += 1


class ChannelFP2QATLinear(nn.Module):
    """QAT 训练专用层：可导的 channel-FP2 fake-quant 前向 + 推理打包导出。

    - 训练（model.train()）时：forward 用 `channel_fake_quantize`（STE），
      权重先经 channel-FP2 round-trip 再做 matmul，梯度经直通回传 `weight`。
    - 推理/导出时：先调用 `quantize()` 生成 `_packed_weight`，forward 走
      CUDA kernel（可用时）或 emu 解码，与 `ChannelFP2Linear` 一致。
    """
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self._packed_weight: Optional[torch.Tensor] = None
        self._channel_scale_index: Optional[torch.Tensor] = None
        # fake-quant 缓存（训练加速，见 _WQ_VERSION 机制）
        self._wq_cache: Optional[torch.Tensor] = None
        self._wq_ver: int = -1

    def quantize(self):
        """按推理层语义打包（与 ChannelFP2Linear.quantize 一致）。"""
        w = self.weight.data.detach().float()
        oc, ic = self.out_features, self.in_features
        nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
        ic_padded = nb_ic * ELEMS_PER_BLOCK
        if ic_padded != ic:
            w_full = torch.zeros(oc, ic_padded, device=w.device, dtype=w.dtype)
            w_full[:, :ic] = w
        else:
            w_full = w
        packed, expv = pack_blockwise_paired(w_full)
        self._packed_weight = packed
        self._channel_scale_index = expv

    force_fake_quant: bool = False   # eval 时也走量化前向（PPL 评测用）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.force_fake_quant:
            # QAT fake-quant：缓存量化权重（累积组内权重不变，直接复用），
            # 用差 trick 保持 STE 语义：前向值 = 量化值，梯度恒等地直通 weight。
            if self._wq_cache is None or self._wq_ver != _WQ_VERSION[0]:
                with torch.no_grad():
                    self._wq_cache = channel_fake_quantize(
                        self.weight.detach()).detach().to(self.weight.dtype)
                self._wq_ver = _WQ_VERSION[0]
            w_eff = self.weight + (self._wq_cache - self.weight).detach()
            return torch.nn.functional.linear(x, w_eff, self.bias)
        # 推理：走 packed 解码（kernel 或 emu）
        if self._packed_weight is not None and self._use_kernel_for(x):
            decoded = _kernel.decode_fp32(self._packed_weight.to(x.device),
                                          self.weight.numel()) if _detect_format(
                self._packed_weight.numel(),
                ((self.weight.numel() + ELEMS_PER_BLOCK - 1)
                 // ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK) == "v1" else \
                _unpack_weight(self._packed_weight,
                               ((self.weight.numel() + ELEMS_PER_BLOCK - 1)
                                // ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK,
                               device=x.device).to(x.device)
            w_decoded = decoded.view_as(self.weight)
        else:
            w_decoded = self.weight
        return torch.nn.functional.linear(x, w_decoded, self.bias)

    def _use_kernel_for(self, x: torch.Tensor) -> bool:
        return (
            _KERNEL_AVAILABLE and _kernel is not None
            and torch.cuda.is_available() and x.is_cuda
        )


def apply_channel_fp2_qat(model: nn.Module) -> nn.Module:
    """把 nn.Linear 替换为 QAT 训练层 ChannelFP2QATLinear（STE fake-quant）。

    用于 QAT 训练（trainer）。训练完成后调用各层 .quantize() 生成 packed，
    再走现有 exporter / loader 导出推理。
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if 'lm_head' not in name and 'embed' not in name.lower():
                qlayer = ChannelFP2QATLinear(
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
            apply_channel_fp2_qat(module)
    return model


def decode_layer_weight(layer: nn.Module, device: str | torch.device = "cpu"):
    """把已量化层的 _packed_weight 解码回 fp32 权重 (用于 loader / 测试)。

    ChannelFP2Linear：按 (oc * ceil(ic/16)) × 16 块序解码，丢弃 pad 元素，
    还原为 (oc, ic)；FP2Linear 直接还原 (out, in)。
    """
    w = layer.weight.data.detach()
    oc, ic = w.shape
    packed = layer._packed_weight
    if packed is None:
        return w.clone()

    nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
    n_total = oc * nb_ic * ELEMS_PER_BLOCK
    decoded = _unpack_weight(packed, n_total, device=device).reshape(oc, -1)
    # pad 丢弃：还原到原始 ic
    return decoded[:, :ic].reshape_as(w)


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


def set_force_fake_quant(model: nn.Module, flag: bool):
    """批量开关所有 QAT 层的 force_fake_quant（eval 前向也走量化语义）。"""
    n = 0
    for m in model.modules():
        if hasattr(m, 'force_fake_quant'):
            m.force_fake_quant = flag
            n += 1
    return n


# ============================================================================
# 混合量化：NVFP4（敏感层）+ SPFP2（MLP 大头）+ BF16（head/状态参数）
# ============================================================================

class NVFP4QATLinear(nn.Module):
    """NVFP4 (E2M1 + per-16-block FP8 E4M3 scale) QAT 训练层。

    前向走 nvfp4_fake_quant（STE 纯直通），缓存机制与 ChannelFP2QATLinear
    一致（累积组内权重不变时复用）。用于混合策略中量化敏感层
    （DeltaNet in/out_proj、attention q/k/v/o 等）。

    注意：导出走 torchao/NVIDIA 工具链的 NVFP4 checkpoint（后续接入），
    本层只负责训练侧 fake-quant。
    """
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        assert _NVFP4_AVAILABLE, "nvfp4_emu 不可用"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self._wq_cache: Optional[torch.Tensor] = None
        self._wq_ver: int = -1

    force_fake_quant: bool = False   # eval 时也走量化前向（PPL 评测用）；
                                     # 曾缺失导致 PPL 读数虚高（154 层未量化）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.force_fake_quant:
            if self._wq_cache is None or self._wq_ver != _WQ_VERSION[0]:
                with torch.no_grad():
                    self._wq_cache = nvfp4_fake_quant(
                        self.weight.detach()).detach().to(self.weight.dtype)
                self._wq_ver = _WQ_VERSION[0]
            w_eff = self.weight + (self._wq_cache - self.weight).detach()
            return torch.nn.functional.linear(x, w_eff, self.bias)
        return torch.nn.functional.linear(x, self.weight, self.bias)


class TwoFourQATLinear(nn.Module):
    """v3 QAT 层：2:4 结构化稀疏 + SPFP2 量化（MLP 层的 v3 升级档）。

    前向 = two_four_fake_quant（量化 → 每组≤2非零约束，STE 直通），
    缓存机制与其它 QAT 层一致。导出用 two_four_pack（1.75 bit/权重）。
    """
    force_fake_quant: bool = False

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        assert _V3_AVAILABLE, "two_four 模块不可用"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self._wq_cache: Optional[torch.Tensor] = None
        self._wq_ver: int = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.force_fake_quant:
            if self._wq_cache is None or self._wq_ver != _WQ_VERSION[0]:
                with torch.no_grad():
                    self._wq_cache = two_four_fake_quant(
                        self.weight.detach()).to(self.weight.dtype)
                self._wq_ver = _WQ_VERSION[0]
            w_eff = self.weight + (self._wq_cache - self.weight).detach()
            return torch.nn.functional.linear(x, w_eff, self.bias)
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def quantize(self):
        """v3 位流导出（1.75 bit/权重）。"""
        self._packed_weight = two_four_pack(self.weight.data.detach())
        self._packed_format = "v3-24"


def _nvfp4_replacement(module: nn.Linear) -> nn.Module:
    q = NVFP4QATLinear(module.in_features, module.out_features,
                       module.bias is not None,
                       device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        q.weight.copy_(module.weight)
        if module.bias is not None:
            q.bias.copy_(module.bias)
    return q


class INT8QATLinear(nn.Module):
    """per-channel 对称 INT8 QAT 层（混合策略中 lm_head 的安全档）。"""
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        assert _INT8_AVAILABLE, "int8_emu 不可用"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self._wq_cache: Optional[torch.Tensor] = None
        self._wq_ver: int = -1

    force_fake_quant: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.force_fake_quant:
            if self._wq_cache is None or self._wq_ver != _WQ_VERSION[0]:
                with torch.no_grad():
                    self._wq_cache = int8_fake_quant(
                        self.weight.detach()).detach().to(self.weight.dtype)
                self._wq_ver = _WQ_VERSION[0]
            w_eff = self.weight + (self._wq_cache - self.weight).detach()
            return torch.nn.functional.linear(x, w_eff, self.bias)
        return torch.nn.functional.linear(x, self.weight, self.bias)


class NVFP4EmbeddingQAT(nn.Module):
    """NVFP4 fake-quant Embedding（混合策略中 embed 档）。

    forward 用量化权重的 straight-through 查表（F.embedding），梯度直通
    回 embedding.weight，缓存机制与 QAT Linear 一致。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx=None, device=None, dtype=None):
        super().__init__()
        assert _NVFP4_AVAILABLE
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim,
                                               device=device, dtype=dtype))
        self._wq_cache: Optional[torch.Tensor] = None
        self._wq_ver: int = -1

    force_fake_quant: bool = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.training or self.force_fake_quant:
            if self._wq_cache is None or self._wq_ver != _WQ_VERSION[0]:
                with torch.no_grad():
                    self._wq_cache = nvfp4_fake_quant(
                        w.detach()).detach().to(self.weight.dtype)
                self._wq_ver = _WQ_VERSION[0]
            w = self.weight + (self._wq_cache - self.weight).detach()
        return torch.nn.functional.embedding(
            input_ids, w, padding_idx=self.padding_idx)


def _spfp2_replacement(module: nn.Linear) -> nn.Module:
    q = ChannelFP2QATLinear(module.in_features, module.out_features,
                            module.bias is not None,
                            device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        q.weight.copy_(module.weight)
        if module.bias is not None:
            q.bias.copy_(module.bias)
    return q


def apply_mixed_quant(model: nn.Module, quantize_head: bool = False,
                      v3: bool = False, fp3_tier: bool = False,
                      fp3_aggressive: bool = False) -> nn.Module:
    """混合量化替换（NVFP4 + SPFP2 + INT8 + BF16）：

      - 默认（quantize_head=False）：
          BF16 保留 embed / lm_head（及所有非 Linear 状态参数如 DeltaNet
          的 A_log/dt_bias —— 天然不是 nn.Linear，不会被触碰）
          NVFP4：attention / DeltaNet / gate 类投影（量化敏感）
          SPFP2：MLP gate/up/down_proj（参数大头，保压缩率）
      - quantize_head=True（9B 级 ~4.3GB 用）：
          embed → NVFP4（查表冗余度高，低风险）
          lm_head → 保持 BF16 参与训练（零量化噪声/全精度梯度），
          导出时后置 PTQ 转 INT8 per-channel（实测 round-trip 误差 ~1.1%，
          贪心解码 4/4 逐 token 一致，见 int8_emu.ptq_int8_pack）
          tied embeddings 时 lm_head 与 QAT embed 共享 weight

    返回 model，并打印各档位层数/参数量统计。
    """
    stats = {"nvfp4": [0, 0], "spfp2": [0, 0], "int8": [0, 0], "bf16": [0, 0]}

    # tied embeddings 检测：weight 同一块存储 → lm_head 跳过
    embed_weight_ref = None
    embed_path = None
    for path, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embed_weight_ref = module.weight
            embed_path = path
            break

    def _is_tied_head(module) -> bool:
        return (embed_weight_ref is not None
                and module.weight is embed_weight_ref)

    tied_head_path = None

    # 先收集再替换（避免迭代中修改结构）
    replacements = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if quantize_head:
                replacements.append((path, "nvfp4_embed"))
                stats["nvfp4"][0] += 1
                stats["nvfp4"][1] += module.weight.numel()
            else:
                stats["bf16"][0] += 1
                stats["bf16"][1] += module.weight.numel()
            continue
        if not isinstance(module, torch.nn.Linear):
            # 非 Linear 的敏感参数（A_log/dt_bias）显式计入 BF16 档
            for pn, p in module.named_parameters(recurse=False):
                if 'A_log' in pn or 'dt_bias' in pn:
                    stats["bf16"][1] += p.numel()
            continue
        if 'lm_head' in path:
            # lm_head 始终 BF16 参与训练；导出时后置 PTQ 转 INT8（方案定稿）。
            # INT8QATLinear 保留为两阶段收尾的可选项，默认不启用。
            # tied 且 embed 将被量化时：记录路径，embed 替换后共享 weight
            # （tie 下 head 必然跟随 embed 的 NVFP4 档）。
            if quantize_head and _is_tied_head(module):
                tied_head_path = path
                stats["nvfp4"][0] += 1
                stats["nvfp4"][1] += module.weight.numel()
            else:
                stats["bf16"][0] += 1
                stats["bf16"][1] += module.weight.numel()
            continue
        if 'embed' in path.lower():
            stats["bf16"][0] += 1
            stats["bf16"][1] += module.weight.numel()
            continue
        if 'mlp' in path:
            kind = "twofour" if v3 else "spfp2"
            replacements.append((path, kind))
            stats["spfp2"][0] += 1
            stats["spfp2"][1] += module.weight.numel()
        elif fp3_tier and fp3_aggressive and 'o_proj' in path:
            # 实验2(激进): 仅 attn o_proj 用 FP3，其余全降 SPFP2
            kind = "fp3"
            replacements.append((path, kind))
            stats.setdefault("fp3", [0, 0])
            stats["fp3"][0] += 1
            stats["fp3"][1] += module.weight.numel()
        elif fp3_tier and fp3_aggressive and 'out_proj' not in path:
            # 实验2(激进): MLP+GDN门控+GDN主投影+attn q/k/v 全部 SPFP2
            kind = "spfp2"
            replacements.append((path, kind))
            stats["spfp2"][0] += 1
            stats["spfp2"][1] += module.weight.numel()
        elif fp3_tier and 'in_proj_ba' in path:
            # 实验1: GDN 门控 (b/a) 降为 SPFP2
            kind = "spfp2"
            replacements.append((path, kind))
            stats["spfp2"][0] += 1
            stats["spfp2"][1] += module.weight.numel()
        elif fp3_tier and 'out_proj' not in path:
            # 实验1: 其余 attn/GDN 用 FP3 (3.5bit)
            kind = "fp3"
            replacements.append((path, kind))
            stats.setdefault("fp3", [0, 0])
            stats["fp3"][0] += 1
            stats["fp3"][1] += module.weight.numel()
        else:
            kind = "nvfp4"
            replacements.append((path, kind))
            stats["nvfp4"][0] += 1
            stats["nvfp4"][1] += module.weight.numel()

    qat_embed_module = None
    for path, kind in replacements:
        parent = model.get_submodule('.'.join(path.split('.')[:-1]))
        leaf = path.split('.')[-1]
        old = getattr(parent, leaf)
        if kind == "spfp2":
            setattr(parent, leaf, _spfp2_replacement(old))
        elif kind == "fp3":
            q = FP3QATLinear(old.in_features, old.out_features,
                             old.bias is not None,
                             device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                q.weight.copy_(old.weight)
                if old.bias is not None:
                    q.bias.copy_(old.bias)
            setattr(parent, leaf, q)
        elif kind == "twofour":
            q = TwoFourQATLinear(old.in_features, old.out_features,
                                old.bias is not None,
                                device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                q.weight.copy_(old.weight)
                if old.bias is not None:
                    q.bias.copy_(old.bias)
            setattr(parent, leaf, q)
        elif kind == "nvfp4":
            setattr(parent, leaf, _nvfp4_replacement(old))
        elif kind == "int8":
            q = INT8QATLinear(old.in_features, old.out_features,
                              old.bias is not None,
                              device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                q.weight.copy_(old.weight)
                if old.bias is not None:
                    q.bias.copy_(old.bias)
            setattr(parent, leaf, q)
        elif kind == "nvfp4_embed":
            qe = NVFP4EmbeddingQAT(old.num_embeddings, old.embedding_dim,
                                   padding_idx=old.padding_idx,
                                   device=old.weight.device,
                                   dtype=old.weight.dtype)
            with torch.no_grad():
                qe.weight.copy_(old.weight)
            setattr(parent, leaf, qe)
            if path == embed_path:
                qat_embed_module = qe

    # tied head：与 QAT embed 共享 weight（同 NVFP4 档）
    if tied_head_path is not None and qat_embed_module is not None:
        parent = model.get_submodule('.'.join(tied_head_path.split('.')[:-1]))
        leaf = tied_head_path.split('.')[-1]
        old = getattr(parent, leaf)
        head_q = NVFP4QATLinear(old.in_features, old.out_features,
                                old.bias is not None,
                                device=qat_embed_module.weight.device,
                                dtype=qat_embed_module.weight.dtype)
        head_q.weight = qat_embed_module.weight      # 共享 Parameter，保持 tie
        if old.bias is not None:
            with torch.no_grad():
                head_q.bias.copy_(old.bias)
        setattr(parent, leaf, head_q)

    tot = sum(v[1] for v in stats.values()) or 1
    for k, (n, p) in stats.items():
        if n or p:
            print(f"[SPARK-MIX] {k.upper():5s}: {n:4d} 层  {p/1e9:.2f}B ({p/tot*100:.1f}%)")
    avg_bits = (stats['spfp2'][1]*2.5 + stats['nvfp4'][1]*4.5 +
                stats['int8'][1]*8 + stats['bf16'][1]*16) / tot
    est_gb = (stats['spfp2'][1]*2.5 + stats['nvfp4'][1]*4.5 +
              stats['int8'][1]*8 + stats['bf16'][1]*16) / 8 / 1e9
    print(f"[SPARK-MIX] 综合密度 ≈ {avg_bits:.2f} bit/权重, 预计权重体积 ≈ {est_gb:.2f} GB")
    return model
