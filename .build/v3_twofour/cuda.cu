#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
/**
 * SPARK v3 — 2:4 Sparse + 1.75-bit TwoFour 在线反量化 CUDA Kernel
 *
 * 格式: 双块 32 权重 = 7 字节 (56 bit)
 *   [mode0-3(16bit)] [sign0-3(8bit)] [exp0(4)|exp1(4)] [mode4-7] [sign4-7] [pad]
 *   mode: 0-5 双非零, 8-11 单非零, 15 全零
 *   密度: 1.75 bit/权重 (vs SPFP2 的 2.5 bit, -30%)
 *
 * 解码后可直接喂 tensor core（2:4 结构化稀疏，Ampere+ 原生支持）
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// TwoFour 解码（inline device，GEMM prologue 用）
// ============================================================

__device__ __forceinline__ void decode_twofour_block(
    const uint8_t* __restrict__ packed,  // 7 bytes per pair (32 weights)
    float* out,                           // 32 decoded values
    int pair_idx)
{
    const uint8_t* p = packed + pair_idx * 7;
    uint64_t v = 0;
    #pragma unroll
    for (int k = 0; k < 7; ++k)
        v |= ((uint64_t)p[k]) << (8 * k);

    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        int base = slot * 28;
        uint8_t exp_raw = (v >> (base + 24)) & 0xF;
        float scale = exp2f((float)exp_raw - 10.0f);

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint8_t mode = (v >> (base + 4 * g)) & 0xF;
            uint8_t s0 = (v >> (base + 16 + 2 * g)) & 1;
            uint8_t s1 = (v >> (base + 16 + 2 * g + 1)) & 1;

            float* dst = out + slot * 16 + g * 4;
            dst[0] = dst[1] = dst[2] = dst[3] = 0.0f;

            // Pattern table for dual-nonzero modes 0-5
            // C(4,2) = 6 patterns: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
            if (mode < 6) {
                // Unrolled pattern switch for performance
                int a, b;
                switch (mode) {
                    case 0: a = 0; b = 1; break;
                    case 1: a = 0; b = 2; break;
                    case 2: a = 0; b = 3; break;
                    case 3: a = 1; b = 2; break;
                    case 4: a = 1; b = 3; break;
                    default: a = 2; b = 3; break;
                }
                dst[a] = s0 ? -scale : scale;
                dst[b] = s1 ? -scale : scale;
            } else if (mode >= 8 && mode <= 11) {
                // Single nonzero
                int a = mode - 8;
                dst[a] = s0 ? -scale : scale;
            }
            // mode == 15: all zero (already initialized)
        }
    }
}

// ============================================================
// 2:4 Sparse metadata 生成（用于 tensor core 的 sparse 指令）
// ============================================================

// NVIDIA 2:4 sparse format requires a metadata tensor that describes
// which 2 of every 4 weights are non-zero. The metadata is used by
// the tensor core to skip zero values.

__device__ __forceinline__ uint16_t generate_sparse_metadata(
    const uint8_t* __restrict__ packed,
    int pair_idx)
{
    // For each group of 4 weights, 2 bits of metadata:
    //   bit pattern indicates which 2 positions are non-zero
    // This maps to the tensor core's sparse format
    const uint8_t* p = packed + pair_idx * 7;
    uint64_t v = 0;
    for (int k = 0; k < 7; ++k)
        v |= ((uint64_t)p[k]) << (8 * k);

    uint16_t meta = 0;
    // 8 groups per pair, 2 bits each = 16 bits
    for (int slot = 0; slot < 2; ++slot) {
        int base = slot * 28;
        for (int g = 0; g < 4; ++g) {
            uint8_t mode = (v >> (base + 4 * g)) & 0xF;
            uint8_t m2;
            if (mode < 6) {
                // Dual nonzero: encode position pair
                // Tensor core expects specific encoding
                switch (mode) {
                    case 0: m2 = 0x3; break;  // positions 0,1
                    case 1: m2 = 0x5; break;  // positions 0,2
                    case 2: m2 = 0x9; break;  // positions 0,3
                    case 3: m2 = 0x6; break;  // positions 1,2
                    case 4: m2 = 0xA; break;  // positions 1,3
                    default: m2 = 0xC; break; // positions 2,3
                }
            } else {
                m2 = 0x0;  // sparse or zero (tensor core treats as dense zeros)
            }
            int gi = slot * 4 + g;
            meta |= (m2 << (gi * 2));
        }
    }
    return meta;
}

// ============================================================
// Naive GEMM with TwoFour decode (概念验证)
// ============================================================

__global__ void twofour_dequant_gemm_naive(
    const uint8_t* __restrict__ w_packed,     // [oc, packed_bytes_per_row]
    const half*  __restrict__ x,              // [batch_seq, ic]
    half*        __restrict__ y,              // [batch_seq, oc]
    const float* __restrict__ bias,           // [oc] or null
    int batch_seq, int oc, int ic,
    int pairs_per_row)                        // ic / 32
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_seq || col >= oc) return;

    float acc = 0.0f;
    float wv[32];

    for (int pair = 0; pair < pairs_per_row; ++pair) {
        decode_twofour_block(w_packed + col * pairs_per_row * 7,
                             wv, pair);

        int ic_base = pair * 32;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            int ic_idx = ic_base + j;
            if (ic_idx < ic) {
                acc += wv[j] * __half2float(x[row * ic + ic_idx]);
            }
        }
    }

    if (bias) acc += bias[col];
    y[row * oc + col] = __float2half_rn(acc);
}

// ============================================================
// PyTorch tensor 接口
// ============================================================

void spark_v3_twofour_forward(
    torch::Tensor w_packed,     // uint8 [total_packed_bytes]
    torch::Tensor x,            // half [batch_seq, ic]
    torch::Tensor y,            // half [batch_seq, oc]
    torch::Tensor bias,         // float [oc] or empty
    int64_t oc, int64_t ic)
{
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int batch_seq = x.size(0);
    int pairs_per_row = (ic + 31) / 32;
    dim3 block(16, 16);
    dim3 grid((oc + 15) / 16, (batch_seq + 15) / 16);
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    twofour_dequant_gemm_naive<<<grid, block, 0, stream>>>(
        (const uint8_t*)w_packed.data_ptr(),
        (const half*)x.data_ptr(),
        (half*)y.data_ptr(),
        bias_ptr,
        (int)batch_seq, (int)oc, (int)ic, pairs_per_row);
}
