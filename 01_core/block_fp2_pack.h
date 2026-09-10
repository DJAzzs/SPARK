// SPARK Block-wise Shared-Exponent FP2 — 位布局单一事实源 (single source of truth)
//
//   ┌─────────────────────────────── block layout (5 bytes = 40 bits) ──────────────┐
//   │ bit       0..31 : 16 × 2bit mantissa codes (element i in [2i, 2i+1))          │
//   │ bit      32..35 : shared 4-bit FP4 exponent                                   │
//   │ bit      36..39 : padding == 0                                                │
//   └───────────────────────────────────────────────────────────────────────────────┘
//
// NOTE: this header is the C++/CUDA authority. The PyTorch reference emu
// (block_fp2_emu.py) mirrors these exact constants; bit-exact agreement between
// emu and kernel is enforced by 05_tests/test_block_decoding.py.
#pragma once

#define SPARK_ELEMS_PER_BLOCK  16      // weights sharing one exponent
#define SPARK_MANTISSA_BITS    2       // bits per element mantissa (codes {-1,0,+1})
#define SPARK_EXP_BITS         4       // shared FP4 exponent width
#define SPARK_BYTES_PER_BLOCK  5       // (16*2 + 4) = 36 bit -> aligned to 40 bit

// decode: value = mantissa * 2^(exp - kExpBias)
static __device__ __host__ inline float spark_exp_bias() { return 2.0f; }

// ---- unpack a single block (byte pointer b of length SPARK_BYTES_PER_BLOCK) -----
template <typename T>
__device__ __host__ inline void spark_unpack_block(
        const unsigned char* b,    // in : 5 bytes little-endian packed block
        T* out,                    // out: [SPARK_ELEMS_PER_BLOCK] decoded values (fp32/fp16)
        int /*unused*/ = 0) {
    // assemble bits 0..31 as a uint32 LE word from b[0..3]
    unsigned int mant =
        ((unsigned int)b[0]) |
        ((unsigned int)b[1] <<  8) |
        ((unsigned int)b[2] << 16) |
        ((unsigned int)b[3] << 24);
#if defined(__CUDA_ARCH__)
    const float scale = __exp2f((float)((int)(b[4] & 0x0Fu)) - spark_exp_bias());
#else
    // host path: use standard math
    #include <cmath>
    const int expv = (int)(b[4] & 0x0Fu);
    const float scale = std::exp2f((float)expv - spark_exp_bias());
#endif

#pragma unroll
    for (int i = 0; i < SPARK_ELEMS_PER_BLOCK; ++i) {
        unsigned int code = (mant >> (SPARK_MANTISSA_BITS * i)) & 3u;
#if defined(__CUDA_ARCH__)
        float v;
#else
        float v = 0.0f;
#endif
        switch (code) {            // mantissa map {00:+1, 01:-1, 10:0, 11:0}
            case 0u: v =  1.0f; break;
            case 1u: v = -1.0f; break;
            default: v =  0.0f; break;
        }
#if defined(__CUDA_ARCH__)
        out[i] = __float2half_rn(v * scale);
#else
        out[i] = static_cast<T>(v * scale);
#endif
    }
}

// ---- pack a single block from fp32 source into 5 bytes --------------------------
template <typename T>
__device__ __host__ inline void spark_pack_block(
        const T* in,               // in : [SPARK_ELEMS_PER_BLOCK] values (fp32/fp16)
        unsigned char* b /* out: 5 bytes */) {
    float maxv = 0.0f;
#pragma unroll
    for (int i = 0; i < SPARK_ELEMS_PER_BLOCK; ++i) {
#if defined(__CUDA_ARCH__)
        float a = __fabsf((float)in[i]);
#else
        float a = std::fabs(static_cast<float>(in[i]));
#endif
        maxv = fmaxf(maxv, a);
    }
    // choose exponent s.t. scale covers the block max; exp in [0,15]
    int expv;
    if (maxv == 0.0f) {
        expv = 0;
    } else {
#if defined(__CUDA_ARCH__)
        float l2 = __log2f(maxv);
#else
        float l2 = std::log2f(maxv);
#endif
        expv = (int)(l2 + spark_exp_bias() - 1.0f + 0.5f);   // round
    }
    if (expv < 0)   expv = 0;
    if (expv > 15)  expv = 15;
#if defined(__CUDA_ARCH__)
    float scale = __exp2f((float)expv - spark_exp_bias());
#else
    float scale = std::exp2f((float)expv - spark_exp_bias());
#endif

    unsigned int mant = 0u;
#pragma unroll
    for (int i = 0; i < SPARK_ELEMS_PER_BLOCK; ++i) {
#if defined(__CUDA_ARCH__)
        float x   = (float)in[i];
#else
        float x   = static_cast<float>(in[i]);
#endif
        unsigned int code;
        if (x > scale * 0.5f)       code = 0u;          // +1
        else if (x < -scale*0.5f)   code = 1u;          // -1
        else                        code = 2u;          //  0
        mant |= code << (SPARK_MANTISSA_BITS * i);
    }
    b[0] = (unsigned char)( mant        & 0xFFu);
    b[1] = (unsigned char)((mant >>  8) & 0xFFu);
    b[2] = (unsigned char)((mant >> 16) & 0xFFu);
    b[3] = (unsigned char)((mant >> 24) & 0xFFu);
    b[4] = (unsigned char)(expv);                       // low nibble used
}
