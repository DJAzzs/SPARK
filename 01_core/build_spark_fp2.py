"""SPARK CUDA 扩展编译入口（缓存，避免重复编译）。

.device kernel 在 block_fp2_kernel.cu；host/pybind 绑定用 inline C++，
经 torch.utils.cpp_extension.load_inline 合并编译为模块 spark_fp2_ext。
提供：decode_fp32(packed, rows) / decode_fp16(packed, rows)。
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0;12.1+PTX")

_KERNEL_DIR = Path(__file__).resolve().parent
_PACK_H     = _KERNEL_DIR / "block_fp2_pack.h"

_MODULE_NAME = "spark_fp2_ext"
_cache       = {"mod": None}

_INLINE_HOST_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// forward declarations of extern-C launchers defined in block_fp2_kernel.cu
extern "C" void spark_decode_launch_f32(const void*, void*, int, cudaStream_t);
extern "C" void spark_decode_launch_f16(const void*, void*, int, cudaStream_t);

torch::Tensor decode_generic(torch::Tensor packed, int64_t rows,
                             bool half) {
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "expected uint8 bytes");
    const long nb = (long)(packed.numel() / 5);          // SPARK_BYTES_PER_BLOCK=5
    const long total = nb * 16L;                         // SPARK_ELEMS_PER_BLOCK
    TORCH_CHECK(total == rows, "decode size mismatch");
    auto out = torch::empty({rows},
        packed.options().dtype(half ? torch::kFloat16 : torch::kFloat32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (half) {
        spark_decode_launch_f16(packed.data_ptr(), out.data_ptr(),
                                (int)nb, stream);
    } else {
        spark_decode_launch_f32(packed.data_ptr(), out.data_ptr(),
                                (int)nb, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor decode_fp32(torch::Tensor packed, int64_t rows) {
    return decode_generic(std::move(packed), rows, false);
}
torch::Tensor decode_fp16(torch::Tensor packed, int64_t rows) {
    return decode_generic(std::move(packed), rows, true);
}

// 注意：绑定由 load_inline(functions=[...]) 自动生成，此处不要手写 PYBIND11_MODULE
"""


def load_kernel(force_rebuild: bool = False):
    if not force_rebuild and _cache["mod"] is not None:
        return _cache["mod"]

    import torch                       # 先加载 libtorch，.so 才能解析符号
    assert (_PACK_H.exists()), f"missing {_PACK_H}"
    from torch.utils import cpp_extension as ce

    kernel_path = _KERNEL_DIR / "block_fp2_kernel.cu"
    assert kernel_path.exists(), f"missing {kernel_path}"
    build_dir = os.path.join(os.environ.get("SPARK_BUILD_DIR",
                                            "/tmp/opencode/spark_ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    inc = f"-I{_KERNEL_DIR}"
    try:
        mod = ce.load_inline(
            name=_MODULE_NAME,
            cpp_sources=_INLINE_HOST_CPP,
            cuda_sources=[kernel_path.read_text()],
            functions=["decode_fp32", "decode_fp16"],
            extra_cflags=["-O3"],                 # torch 2.15 nightly: C++20 由加载器提供
            extra_cuda_cflags=["-O3", inc],       # 让 kernel 能 #include <block_fp2_pack.h>
            build_directory=build_dir,
            with_cuda=True,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"SPARK CUDA 扩展编译失败：{e}\n"
            f"(kernel: {kernel_path}；请确认 nvcc>=12 / sm_120a)") from e
    _cache["mod"] = mod
    return mod


if __name__ == "__main__":
    m = load_kernel()
    print("SPARK FP2 extension OK:", [x for x in dir(m) if x.startswith("decode")])
