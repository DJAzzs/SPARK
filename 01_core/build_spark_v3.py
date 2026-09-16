"""SPARK v3 CUDA 扩展加载器"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0;12.1+PTX")

_DIR = Path(__file__).resolve().parent
_cu = _DIR / "spark_v3_kernel.cu"
_cache = {"mod": None}


def load_v3_kernel(force=False):
    if not force and _cache["mod"] is not None:
        return _cache["mod"]
    import torch
    from torch.utils import cpp_extension as ce
    mod = ce.load_inline(
        name="spark_v3_ext",
        cpp_sources="""
#include <torch/extension.h>
void spark_v3_spfp2_forward(torch::Tensor w, torch::Tensor x,
    torch::Tensor y, torch::Tensor b, int64_t oc, int64_t ic);
""",
        cuda_sources=[_cu.read_text()],
        functions=["spark_v3_spfp2_forward"],
        extra_cuda_cflags=["-O3", f"-I{_DIR}"],
        build_directory=os.environ.get("SPARK_BUILD_DIR",
                                        str(_DIR.parent / ".build" / "v3")),
        with_cuda=True,
        verbose=False,
    )
    _cache["mod"] = mod
    return mod


if __name__ == "__main__":
    m = load_v3_kernel()
    print("v3 kernel OK:", dir(m))
