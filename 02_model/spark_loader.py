"""SPARK 权重加载器"""
from __future__ import annotations

import os, sys, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "01_core") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "01_core"))


class SparkWeightLoader:
    """从 .spark 文件还原量化权重。"""
    def __init__(self, spark_dir: str):
        self.spark_dir = spark_dir
        pth = os.path.join(spark_dir, "state_dict.pt")
        assert os.path.exists(pth), f"not found: {pth}"
        self.data = torch.load(pth, map_location="cpu", weights_only=False)
        print(f"[SPARK-LOADER] loaded {len(self.data)} tensors from .spark")

    def state_dict(self):
        """返回解码后的 state_dict（FP16/BF16 用于推理）。"""
        from quant_linear import ChannelFP2Linear, FP2Linear, decode_layer_weight
        state = {}

        # 先用 _scale_index 重建形状提示：_scale_index 形状 = (oc * nb_ic) 或块数
        # 但 decoder 需要知道 (oc, ic)。从包尺寸推断：n_blocks = numel//5，每行块数未知。
        # 更稳妥: 打包时同时把 (oc, ic) 维度元数据存进 state_dict（见 exporter）。
        for name in list(self.data.keys()):
            if name.endswith('_meta'):
                meta = self.data[name]
                pass  # 可选元数据，见 exporter

        # 逐层解码：packed tensor + 其相邻 _scale_index
        packed_keys = sorted(k for k in self.data.keys() if k.endswith('_packed'))
        for pk in packed_keys:
            layer_name = pk.replace('_packed', '')
            packed = self.data[pk]
            # 从 meta 恢复形状（exporter 会写 *_meta = (oc, ic)）
            meta_key = layer_name + '_meta'
            if meta_key in self.data:
                oc, ic = int(self.data[meta_key][0]), int(self.data[meta_key][1])
            else:
                # fallback: 无法得知精确 (oc, ic)，从包大小推 ic 倍数。这里由调用方
                # 提供 shape，见 _decode_with_shape。
                raise KeyError(
                    f"missing '{meta_key}' shape metadata (oc, ic) for layer "
                    f"'{layer_name}' — 请用新版 exporter 重新导出")
            decoded = self._decode_weight(packed, oc, ic)
            # key 对齐 HF state_dict：量化层的权重名带 `.weight` 后缀
            state[layer_name + '.weight'] = decoded
        return state

    def _decode_weight(self, packed: torch.Tensor, oc: int, ic: int) -> torch.Tensor:
        """从 packed uint8 解码为 fp16 weight tensor，shape (oc, ic)。
        自动兼容 v1 (5字节/块) 与 v2 (9字节/双块) 格式。"""
        from quant_linear import _unpack_weight, ELEMS_PER_BLOCK
        nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
        n_total = oc * nb_ic * ELEMS_PER_BLOCK
        decoded = _unpack_weight(packed, n_total,
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
        # 还原 (oc, ic)，丢弃 pad
        return decoded.reshape(oc, -1)[:, :ic].contiguous().to(torch.float16)


def load_spark_model(spark_dir: str, hf_base: str):
    """从 .spark 目录加载 FP2 量化模型进行推理。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[LOAD] HF base: {hf_base}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_base, device_map="cpu", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)

    loader = SparkWeightLoader(spark_dir)
    decoded_state = loader.state_dict()

    model_state = model.state_dict()
    matched = 0
    for name, weight in decoded_state.items():
        if name in model_state and model_state[name].shape == weight.shape:
            with torch.no_grad():
                model_state[name].copy_(weight.to(model_state[name].dtype))
            matched += 1

    print(f"[LOAD] matched {matched}/{len(model_state)} weights")

    tokenizer = AutoTokenizer.from_pretrained(hf_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model.eval(), tokenizer


if __name__ == '__main__':
    loader = SparkWeightLoader("/home/dja/桌面/SPARK/data/spark_checkpoint")
    sd = loader.state_dict()
    print("state dict keys:", list(sd.keys())[:5])


def apply_spark_state(model, state):
    """把 SPARK checkpoint 完整应用到模型（三种条目）：
    - <layer>_packed (+_meta):   SPFP2 v1/v2 解码写回
    - <layer>_nvfp4_weight:      NVFP4 量化值直接写回
    - param::<name>:             非量化参数（norm/lm_head/A_log 等）
    返回 (n_packed, n_nvfp4, n_param)。"""
    from quant_linear import _unpack_weight, ELEMS_PER_BLOCK
    msd = model.state_dict()
    n1 = n2 = n3 = 0
    for key in list(state.keys()):
        if key.endswith('_packed'):
            layer = key[: -len('_packed')]
            meta = state.get(layer + '_meta')
            if meta is None:
                continue
            oc, ic = int(meta[0]), int(meta[1])
            nb_ic = (ic + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK
            dec = _unpack_weight(state[key], oc * nb_ic * ELEMS_PER_BLOCK)
            dec = dec.reshape(oc, -1)[:, :ic]
            sd = layer + '.weight'
            if sd in msd and msd[sd].shape == dec.shape:
                msd[sd].copy_(dec.to(msd[sd].dtype))
                n1 += 1
        elif key.endswith('_nvfp4_weight'):
            layer = key[: -len('_nvfp4_weight')]
            sd = layer + '.weight'
            w = state[key].float()
            if sd in msd and msd[sd].shape == w.shape:
                msd[sd].copy_(w.to(msd[sd].dtype))
                n2 += 1
        elif key.startswith('param::'):
            sd = key[len('param::'):]
            if sd in msd and msd[sd].shape == state[key].shape:
                msd[sd].copy_(state[key].float().to(msd[sd].dtype))
                n3 += 1
    return n1, n2, n3
