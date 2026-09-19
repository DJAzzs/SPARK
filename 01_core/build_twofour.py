"""SPARK v3 TwoFour (2:4 sparse 1.75-bit) CUDA 扩展加载器"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0;12.1+PTX")

_DIR = Path(__file__).resolve().parent
_cu = _DIR / "spark_v3_twofour.cu"
_cache = {"mod": None}


def load_twofour_kernel(force=False):
    if not force and _cache["mod"] is not None:
        return _cache["mod"]
    import torch
    from torch.utils import cpp_extension as ce

    build_dir = str(_DIR.parent / ".build" / "v3_twofour")
    os.makedirs(build_dir, exist_ok=True)

    mod = ce.load_inline(
        name="spark_v3_twofour_ext",
        cpp_sources="""
#include <torch/extension.h>
void spark_v3_twofour_forward(torch::Tensor w, torch::Tensor x,
    torch::Tensor y, torch::Tensor b, int64_t oc, int64_t ic);
""",
        cuda_sources=[_cu.read_text()],
        functions=["spark_v3_twofour_forward"],
        extra_cuda_cflags=["-O3", f"-I{_DIR}"],
        build_directory=build_dir,
        with_cuda=True,
        verbose=False,
    )
    _cache["mod"] = mod
    return mod


if __name__ == "__main__":
    m = load_twofour_kernel()
    print("TwoFour kernel OK:", [x for x in dir(m) if not x.startswith('_')])
