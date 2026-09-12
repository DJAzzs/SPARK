from __future__ import annotations
import os, sys, torch, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _is_spark_quant_layer(module):
    """是否是 SPARK 量化层（有 _packed_weight，可能是 ChannelFP2Linear/FP2Linear）。"""
    return isinstance(module, torch.nn.Module) and hasattr(module, "_packed_weight")


def export_spark_model(model: torch.nn.Module, out_dir: str,
                       base_model_dir: str | None = None):
    """导出量化模型的 .spark 状态（state_dict.pt）。

    对每个 ChannelFP2Linear/FP2Linear，保存：
        <name>_packed  : uint8 打包权重
        <name>_scale_index : 块指数
        <name>_meta    : (oc, ic) 形状元数据（loader 解码所需）
    """
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for name, module in model.named_modules():
        if _is_spark_quant_layer(module):
            if 'embed' not in name.lower() and 'lm_head' not in name:
                if module._packed_weight is not None:
                    n = name.replace('.weight', '')
                    state[n + '_packed'] = module._packed_weight.cpu()
                    if hasattr(module, '_channel_scale_index') and module._channel_scale_index is not None:
                        state[n + '_scale_index'] = module._channel_scale_index.cpu()
                    # 形状元数据：由 weight 维度反推 (oc, ic)
                    oc, ic = module.weight.shape
                    state[n + '_meta'] = torch.tensor([oc, ic], dtype=torch.long)

    # 复制 base 模型的 config（供 HF 重建结构）
    if base_model_dir is not None:
        src_cfg = os.path.join(base_model_dir, "config.json")
        if os.path.exists(src_cfg):
            shutil.copy(src_cfg, os.path.join(out_dir, "config.json"))

    torch.save(state, os.path.join(out_dir, "state_dict.pt"))
    size_mb = os.path.getsize(os.path.join(out_dir, "state_dict.pt")) / (1024**2)
    print(f"[SPARK EXPORT] {out_dir} ({size_mb:.1f}MB, {len(state)} tensors)")
