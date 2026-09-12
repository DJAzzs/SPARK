#include <cuda_runtime.h>

__global__ void k_decode_f32(const unsigned char* __restrict__ packed,
                             float* __restrict__ out, int n_blocks) {
    const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_blocks) return;
    
    const size_t offset = (size_t)i * 5;  // SPARK_BYTES_PER_BLOCK
    unsigned char b[5];
#pragma unroll
    for (int bi = 0; bi < 5; ++bi)
        b[bi] = __ldg(packed + offset + bi);
    
    unsigned int mant =
        ((unsigned int)b[0]) |
        ((unsigned int)b[1] <<  8) |
        ((unsigned int)b[2] << 16) |
        ((unsigned int)b[3] << 24);
    const int expv = (int)(b[4] & 0x0Fu);
    const float scale = __powf(2.0f, (float)(expv - 10));  // 2^(exp-10), bias=10
    
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        unsigned int code = (mant >> (2 * e)) & 3u;
        float v;
        switch (code) {
            case 0: v =  1.0f; break;
            case 1: v = -1.0f; break;
            default: v =  0.0f; break;
        }
        out[(size_t)i * 16 + e] = v * scale;
    }
}

__global__ void k_decode_f16(const unsigned char* __restrict__ packed,
                             half* __restrict__ out, int n_blocks) {
    const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_blocks) return;
    
    const size_t offset = (size_t)i * 5;
    unsigned char b[5];
#pragma unroll
    for (int bi = 0; bi < 5; ++bi)
        b[bi] = __ldg(packed + offset + bi);
    
    unsigned int mant =
        ((unsigned int)b[0]) |
        ((unsigned int)b[1] <<  8) |
        ((unsigned int)b[2] << 16) |
        ((unsigned int)b[3] << 24);
    const int expv = (int)(b[4] & 0x0Fu);
    const float scale = __powf(2.0f, (float)(expv - 10));
    
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        unsigned int code = (mant >> (2 * e)) & 3u;
        float v;
        switch (code) {
            case 0: v =  1.0f; break;
            case 1: v = -1.0f; break;
            default: v =  0.0f; break;
        }
        out[(size_t)i * 16 + e] = __float2half_rn(v * scale);
    }
}

extern "C" {

void spark_decode_launch_f32(const void* packed, void* out,
                             int n_blocks, cudaStream_t stream) {
    const dim3 threads(256);
    const unsigned nb_grid = (unsigned)((n_blocks + threads.x - 1) / threads.x);
    k_decode_f32<<<nb_grid, threads, 0, stream>>>(
        static_cast<const unsigned char*>(packed),
        reinterpret_cast<float*>(out), n_blocks);
}

void spark_decode_launch_f16(const void* packed, void* out,
                             int n_blocks, cudaStream_t stream) {
    const dim3 threads(256);
    const unsigned nb_grid = (unsigned)((n_blocks + threads.x - 1) / threads.x);
    k_decode_f16<<<nb_grid, threads, 0, stream>>>(
        static_cast<const unsigned char*>(packed),
        reinterpret_cast<half*>(out), n_blocks);
}

} // extern "C"
