"""SPARK v2 容器：存储端压缩，加载端解压为 v1 ckpt 同构 state。

压缩策略（基于真实训练权重的实测压缩率，2026-09-13）：
  - NVFP4 层: bf16 量化值 → E2M1 nibble 码流(4bit/w) + FP8 块 scale + zstd(≈-1%)
              （码流近熵下限，zstd 仅为容器统一性）
  - SPFP2 层: v2 packed(9B/双块, 2.25bit/w) + zstd(实测 -22%)
  - lm_head (untied): BF16 → per-channel INT8 (8bit/w)
  - 其余参数(norm/A_log等): fp32 + zstd（保 bit 级 roundtrip）

容器布局（目录）:
  <out>/
    manifest.json     条目索引（类型/形状/压缩/blob 引用）
    tensors/*.zst     每张量一个 zstd blob
    config.json       base 模型 config（如有）

体积（实测口径）: 4B ≈1.95GB(-10%), 9B ≈3.8GB。NVFP4 码流是熵下限，
更小体积需降码率（精度倒退），不在"PPL 持平"约束内。
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "01_core"),
           os.path.join(_PROJECT_ROOT, "02_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import zstandard as zstd

CONTAINER_VERSION = "2.0"


def _compress(t: torch.Tensor) -> bytes:
    if t.dtype == torch.float8_e4m3fn:
        raw = t.view(torch.uint8).numpy().tobytes()
    elif t.dtype == torch.bfloat16:
        raw = t.contiguous().view(torch.uint16).numpy().tobytes()
    else:
        raw = t.contiguous().numpy().tobytes()
    return zstd.ZstdCompressor(level=19).compress(raw)


def _decompress(buf: bytes, dtype: torch.dtype) -> torch.Tensor:
    """解压并按 dtype 解释（长度由解压后字节数自然决定，不用压缩后大小推算）。"""
    raw = zstd.ZstdDecompressor().decompress(buf)
    if dtype == torch.float8_e4m3fn:
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn).clone()
    if dtype == torch.bfloat16:
        return torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16).clone()
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone()


def _write_blob(out_dir: str, name: str, data: bytes) -> str:
    fname = f"tensors/{name}.zst"
    with open(os.path.join(out_dir, fname), "wb") as f:
        f.write(data)
    return fname


def _read_blob(out_dir: str, fname: str) -> bytes:
    with open(os.path.join(out_dir, fname), "rb") as f:
        return f.read()


def save_spark_v2(state: dict, out_dir: str, base_model_dir: str | None = None):
    """把训练 ckpt state（v1 同构）存为 v2 压缩容器。返回体积统计。"""
    from nvfp4_emu import nvfp4_pack
    from int8_emu import ptq_int8_pack
    import shutil

    os.makedirs(os.path.join(out_dir, "tensors"), exist_ok=True)
    manifest = {"version": CONTAINER_VERSION, "format": "spark-v2", "tensors": {}}
    stats = {"raw": 0, "comp": 0}

    def _emit(name, kind, payload, **extra):
        fn = _write_blob(out_dir, name.replace("::", "_"), payload)
        manifest["tensors"][name] = dict(kind=kind, file=fn,
                                         bytes=len(payload), **extra)
        stats["comp"] += len(payload)

    skip_prefixes = ("_meta", "_scale_index", "_nvfp4_scales")  # 并入对应条目
    for key, val in state.items():
        if any(key.endswith(sfx) for sfx in skip_prefixes):
            continue
        raw_b = val.numel() * val.element_size()
        stats["raw"] += raw_b

        if key.endswith("_nvfp4_codes"):
            layer = key[: -len("_nvfp4_codes")]
            meta = state.get(layer + "_meta")
            sc = state.get(layer + "_nvfp4_scales")
            oc, ic = (int(meta[0]), int(meta[1])) if meta is not None \
                else (None, None)
            _emit(key, "nvfp4_codes", _compress(val), oc=oc, ic=ic)
            if sc is not None:
                _emit(layer + "_nvfp4_scales", "nvfp4_scales", _compress(sc))
        elif key.endswith("_nvfp4_weight"):
            # 旧条目（bf16 值）→ 转码流（注意可能有一个码位的二次量化差）
            layer = key[: -len("_nvfp4_weight")]
            meta = state.get(layer + "_meta")
            oc, ic = (int(meta[0]), int(meta[1])) if meta is not None \
                else tuple(val.shape)
            codes, scales = nvfp4_pack(val.float())
            _emit(key, "nvfp4_codes", _compress(codes), oc=oc, ic=ic,
                  codes_numel=codes.numel())
            _emit(key + "::scales", "nvfp4_scales", _compress(scales),
                  scales_numel=scales.numel())
        elif key.endswith("_packed"):
            layer = key[: -len("_packed")]
            meta = state.get(layer + "_meta")
            oc, ic = (int(meta[0]), int(meta[1])) if meta is not None \
                else (None, None)
            _emit(key, "spfp2_packed", _compress(val), oc=oc, ic=ic)
        elif key.startswith("param::") and "lm_head" in key:
            q, scale = ptq_int8_pack(val.float())
            _emit(key, "int8_head_q", _compress(q), shape=list(val.shape))
            _emit(key + "::scale", "int8_head_scale", _compress(scale))
        elif key.startswith("param::"):
            # fp32 存储：bf16 会引入 ~1.35e-3 相对误差破坏 bit 级 roundtrip；
            # 这些参数（norm/A_log 等）体积小，不值得省那一半
            _emit(key, "param_fp32", _compress(val.float()),
                  shape=list(val.shape))
        else:
            _emit(key, "raw", _compress(val))

    if base_model_dir:
        src = os.path.join(base_model_dir, "config.json")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, "config.json"))

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return stats


def load_spark_v2(out_dir: str) -> dict:
    """加载 v2 容器 → v1 ckpt 同构 state（直接喂 apply_spark_state）。"""
    from nvfp4_emu import nvfp4_unpack
    from int8_emu import ptq_int8_unpack

    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    state = {}
    heads = {}          # int8 q/scale 汇合
    nv_meta = {}        # codes/scales 汇合

    for key, e in manifest["tensors"].items():
        kind = e["kind"]
        if kind == "nvfp4_codes":
            codes = _decompress(_read_blob(out_dir, e["file"]), torch.uint8)
            nv_meta.setdefault(key, {})["codes"] = codes
            nv_meta[key]["oc"], nv_meta[key]["ic"] = e["oc"], e["ic"]
        elif kind == "nvfp4_scales":
            scales = _decompress(_read_blob(out_dir, e["file"]),
                                 torch.float8_e4m3fn)
            # 归并到对应 codes 条目（新旧两种命名都处理）
            base = key.replace("::scales", "").replace(
                "_nvfp4_scales", "_nvfp4_codes")
            nv_meta.setdefault(base, {})["scales"] = scales
        elif kind == "spfp2_packed":
            state[key] = _decompress(_read_blob(out_dir, e["file"]), torch.uint8)
            if e.get("oc"):
                state[key.replace("_packed", "") + "_meta"] = \
                    torch.tensor([e["oc"], e["ic"]], dtype=torch.long)
        elif kind == "int8_head_q":
            heads[key] = {"q": _decompress(_read_blob(out_dir, e["file"]),
                                           torch.int8),
                          "shape": e["shape"]}
        elif kind == "int8_head_scale":
            heads[key.replace("::scale", "")]["scale"] = _decompress(
                _read_blob(out_dir, e["file"]), torch.float32)
        elif kind == "param_fp32":
            state[key] = _decompress(_read_blob(out_dir, e["file"]),
                                     torch.float32)
        else:
            state[key] = _decompress(_read_blob(out_dir, e["file"]),
                                     torch.uint8)

    for key, m in nv_meta.items():
        if "codes" in m and "scales" in m:
            # 直通原生码流（解码归 apply_spark_state，容器只管解压）
            state[key] = m["codes"]
            state[key.replace("_nvfp4_codes", "_nvfp4_scales")] = m["scales"]
            if m.get("oc") is not None:
                state[key.replace("_nvfp4_codes", "") + "_meta"] = \
                    torch.tensor([m["oc"], m["ic"]], dtype=torch.long)
    for key, h in heads.items():
        if "scale" in h:
            state[key] = ptq_int8_unpack(h["q"], h["scale"]).bfloat16()
    return state


def container_size_gb(out_dir: str) -> float:
    tot = 0
    for root, _, files in os.walk(out_dir):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
    return tot / 1024**3
