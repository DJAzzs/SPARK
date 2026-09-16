#include <torch/extension.h>

#include <torch/extension.h>
void spark_v3_spfp2_forward(torch::Tensor w, torch::Tensor x,
    torch::Tensor y, torch::Tensor b, int64_t oc, int64_t ic);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("spark_v3_spfp2_forward", torch::wrap_pybind_function(spark_v3_spfp2_forward), "spark_v3_spfp2_forward");
}