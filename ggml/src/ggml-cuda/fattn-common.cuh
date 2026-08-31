#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// One int32 allocation, self-describing so that the kernel signature is
//
//   [0]                 = C, entries reserved per list  (list stride)
//   [1]                 = O, offset in ints of the list region
//   [O + i*C + s]       = block start offset in TOKENS, s < n_i,
//                         STRICTLY ASCENDING
//
//   i = (sequence*ntiles_x + jt)*n_head + head        <- the QUERY head
//
// Block starts are multiples of the consuming kernel's per-iteration KV step
// unit selection RANKS in, a block is the unit the kernel LOADS in. The
// producer emits the union of the selected pages over blocks, which is a

// Per-query-head indexing means a GQA-group union is expressible without any
// kernel change: the producer writes the same list to all query heads of a
// group. The kernel does not know or care which policy produced the list.
    int n_lists;   // nseq * ntiles_x * n_head
    int stride;    // C
    int list_off;  // O
    int total;     // ints to allocate
};

        const int nseq, const int ntiles_x, const int n_head, const int capacity) {
    l.n_lists  = nseq * ntiles_x * n_head;
    l.stride   = capacity;
    l.total    = l.list_off + l.n_lists * l.stride;
    return l;
}

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

//
// *** PLUMBING PRODUCER, NOT THE SELECTOR. ***
// It writes the DEGENERATE list -- every block, every head, ascending -- which
// byte-identical before the real selector exists, so that §4.K can later be
// proven RED against it (a build that selects but does not gate the loads is
// indistinguishable from this one by output, and distinguishable by bytes
        int * KV_pages_ptr, const int n_lists, const int stride, const int list_off,
        const int n_blocks, const int block_tokens) {
    int * GGML_CUDA_RESTRICT KV_pages = KV_pages_ptr;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        KV_pages[0] = stride;
        KV_pages[1] = list_off;
    }

    const int i = blockIdx.x;
    if (i >= n_lists) {
        return;
    }
    if (threadIdx.x == 0) {
    }

    int * L = KV_pages + list_off + i*stride;
    for (int s = threadIdx.x; s < n_blocks; s += blockDim.x) {
        L[s] = s*block_tokens;
    }
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// ===========================================================================
// ===========================================================================
//
// One block per LIST, i.e. per (sequence, query tile, QUERY head). 24 blocks at
// decode. Everything below is shaped by the determinism contract, so read D1-D4
// before changing any of it.
//
// D2 -- fixed reduction shape, no atomics, no cross-wave arrival order.
//   U_p is computed by ONE thread, entirely: 256 sequential float MACs with no
//   cross-thread combine at all. That is stronger than choosing a well-behaved
//   reduction tree -- there is no float reduction to schedule. Threads are
//   independent across PAGES, never within a page.
//   The only cross-thread combine anywhere is an INTEGER count during the
//   threshold search, and integer addition is exactly associative, so its result
//   cannot depend on arrival order either.
//
// D3 -- exact ties break by ascending page index: the emit walk consumes a tie
//   budget in ascending order, so among equal U_p the lowest page indices win.
//
// D4 -- emission is ascending in PAGE index, never rank order. The walk is a
//   single ascending sweep and rank never touches it. This is what makes §4.Q
//   byte-identity hold by construction at K >= n_pages (measured: it does).
//
// Fail toward attending: an invalid page, an empty page, or a NaN bound yields
// U_p = +INF, which always selects. Every uncertainty in this kernel resolves to
// "attend it" (§7.B, §7.D).

// Order-preserving float <-> uint32. Positive floats already compare correctly
// as integers once the sign bit is set; negatives compare in reverse, so they
// are inverted. NaN cannot reach here (it is mapped to +INF before the search).
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

// §3.5).
//
// WHY THE MASK AND NOT K->ne[1]-1
//
// K->ne[1] is the cache's PADDED length: llama_kv_cache::get_n_kv() returns
// GGML_PAD(used_max_p1, 256), so at the operating point it runs 83 cells past
// the last real token. Those cells are not the query and they are not attended
// -- llama_kv_cache sets them to -INF in this very mask. Measuring the resident
// window from them moves its start UP by ceil(pad/B) pages and costs the window
// exactly `pad` real tokens of coverage (measured: 941 covered against a
//
// The mask is the authoritative per-step statement of which cells this query
// attends, and the last FINITE entry of its row is the query's own cell: a
// causal query always attends itself, and every later cell is masked. Reading
// it here needs no new op_param (op_params are baked into a REUSED graph and
// cannot carry a per-step quantity) and no new graph input.
//
// COST: the masked region is a suffix, so the scan runs BACKWARD in blockDim.x
// chunks and terminates on the first chunk containing a finite entry -- one
// iteration of 256 half loads in the steady state, because the padding is < 256
// cells by construction. The whole-row worst case only occurs if the row is
// entirely masked, which yields -1 -> everything resident -> fail toward
// attending (§7).
//
// DETERMINISM (D2): the only cross-thread combine is an integer max, which is
// exactly associative and commutative, so the result cannot depend on arrival
// order -- the same argument that licenses the integer counts in the threshold
// search. No float reduction is introduced.
        const half * mask_row, const int n_kv, int * s_pos_q) {
    if (threadIdx.x == 0) {
        *s_pos_q = -1;
    }
    __syncthreads();

    if (mask_row == nullptr) {
        // No mask means every cell of the view is a real, attended cell.
        if (threadIdx.x == 0) {
            *s_pos_q = n_kv - 1;
        }
        __syncthreads();
        return *s_pos_q;
    }

    const int nthreads = blockDim.x;
    for (int base = ((n_kv - 1)/nthreads)*nthreads; base >= 0; base -= nthreads) {
        const int j = base + threadIdx.x;
        if (j < n_kv && __half2float(mask_row[j]) > -INFINITY) {
            atomicMax(s_pos_q, j);
        }
        __syncthreads();
        // uniform across the block: every thread read the same value after the
        // barrier, so the loop is exited by all of them together
        if (*s_pos_q >= 0) {
            break;
        }
    }

    return *s_pos_q;
}

        int * KV_pages_ptr, const half * bounds, const char * Q_ptr, const half * mask_row,
        const int n_lists, const int stride, const int list_off,
        const int n_pages, const int n_pages_stride,
        const int n_kv, const int n_channels, const int n_head_kv,
        const int page_size, const int block_tokens, const int n_blocks,
        const int top_k, const int n_sink_pages, const int window,
        const int n_head, const size_t nb01, const size_t nb02, const float scale) {
    int * GGML_CUDA_RESTRICT KV_pages = KV_pages_ptr;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        KV_pages[0] = stride;
        KV_pages[1] = list_off;
    }

    const int i = blockIdx.x;
    if (i >= n_lists) {
        return;
    }
    const int head    = i % n_head;                       // QUERY head
    const int kv_head = head / (n_head / n_head_kv);      // its KV head (GQA)

    extern __shared__ float smem_U[];                     // n_pages floats
    __shared__ int   s_cnt[32];
    __shared__ float s_tau;
    __shared__ int   s_budget;
    __shared__ int   s_pos_q;

    // ---- the resident window, from the TRUE query position (§3.5) ----------
    //
    // Derived here rather than passed in, because the host cannot see it: the
    // only host-visible length is the PADDED n_kv, and the padded value is the
    // are constants from hparams, so they still arrive as parameters.
    //
    // Every block recomputes it. That is deliberate: it is one 256-half read in
    // the steady state, against a KV loop orders of magnitude larger, and it
    // keeps the value on the same side of the launch boundary as the code that
    // consumes it -- no second home, no extra launch, no per-step op_param that
    // a reused graph would serve stale.

    // ---- U_p, one thread per page, no cross-thread combine (D2) -----------
    const float * q = (const float *) (Q_ptr + (size_t) head*nb02);
    const half  * rec_base = bounds + (size_t) kv_head*n_pages_stride*2*n_channels;

    for (int p = threadIdx.x; p < n_pages; p += blockDim.x) {
        // resident and out-of-range pages never compete for a rank
        if (p < n_sink_pages || p >= first_window_page) {
            smem_U[p] = -INFINITY;
            continue;
        }
        const half * kmin = rec_base + (size_t) p*2*n_channels;
        const half * kmax = kmin + n_channels;

        float u = 0.0f;
        for (int c = 0; c < n_channels; ++c) {
            const float qc = q[c]*scale;
            const float a  = qc*__half2float(kmin[c]);
            const float b  = qc*__half2float(kmax[c]);
            u += fmaxf(a, b);
        }
        // An INVALIDATED (§4.O) or never-built record must ATTEND. The §6.3
        // sentinel is the inverted interval k_min = +INF, k_max = -INF, so each
        // channel contributes max(q_c*(+INF), q_c*(-INF)) -- which is +INF for
        // q_c != 0 and NaN for q_c == 0. Both routes have to land on the same
        // value, which is what this line is for; the emit loop below then reads
        // that value as the residency signal, and INFINITY is the only value it
        // can test for.
        smem_U[p] = isnan(u) ? INFINITY : u;
    }
    __syncthreads();

    // ---- threshold search: tau = the top_k-th largest ---------------------
    //
    // Bisection on the SORTABLE BIT PATTERN, not on the float range. Mapping
    // float -> uint32 order-preservingly makes 32 integer halvings find tau
    // EXACTLY, with no dependence on the magnitudes involved; bisecting the
    // float range instead needs an initial max, and computing that max in
    // shared memory is where a race would put a run-to-run difference straight
    // into D1. There is no float reduction here at all -- only integer counts,
    // which are exactly associative and so order-independent (D2).
    uint32_t ulo = 0u, uhi = 0xFFFFFFFFu;
    for (int it = 0; it < 32; ++it) {
        const uint32_t umid = ulo + ((uhi - ulo) >> 1);
        int local = 0;
        for (int p = threadIdx.x; p < n_pages; p += blockDim.x) {
        }
        local = warp_reduce_sum(local);
        if ((threadIdx.x & 31) == 0) { s_cnt[threadIdx.x >> 5] = local; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int tot = 0;
            for (int t = 0; t < (int)((blockDim.x + 31)/32); ++t) { tot += s_cnt[t]; }
            s_budget = tot;
        }
        __syncthreads();
        if (s_budget > top_k) { ulo = umid + 1u; } else { uhi = umid; }
        __syncthreads();
        if (ulo >= uhi) { break; }
    }

    // how many are STRICTLY above tau -- the rest of the quota goes to ties, in
    // ascending page order (D3)
    {
        int local = 0;
        for (int p = threadIdx.x; p < n_pages; p += blockDim.x) {
            local += (smem_U[p] > tau) ? 1 : 0;
        }
        local = warp_reduce_sum(local);
        if ((threadIdx.x & 31) == 0) { s_cnt[threadIdx.x >> 5] = local; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int tot = 0;
            for (int t = 0; t < (int)(blockDim.x + 31)/32; ++t) { tot += s_cnt[t]; }
            s_budget = top_k - tot;          // tie quota, may be <= 0
        }
        __syncthreads();
    }

    // ---- emit, ASCENDING in page index (D4) -------------------------------
    //
    // Serial in one thread: n_pages is at most a few thousand and this runs once
    // per list per step, against a KV loop that is orders of magnitude larger.
    // Serial is also what makes the tie budget and the granule union trivially
    // deterministic -- there is no order to get wrong.
    if (threadIdx.x == 0) {
        int  n_emit   = 0;
        int  budget   = s_budget;
        int  last_blk = -1;
        int * L = KV_pages + list_off + i*stride;

        for (int p = 0; p < n_pages; ++p) {
            bool take = false;
            //
            // An INVALIDATED page (§6.3's inverted-interval sentinel, written by
            // through the isnan force-attend line. That alone would leave it
            // COMPETING for the top_k budget: once more than top_k pages carry
            // the sentinel, the bisection returns tau = +INF, `smem_U[p] > tau`
            // is false for every one of them, and the tie quota admits exactly
            // top_k in ascending page order -- the rest silently dropped, which
            // is §4.O violated by its own implementation, in its unsafe
            // direction. So the invalid page joins the RESIDENT set, budget-
            // exempt, exactly as sink and window do. One comparison against a
            // shared-memory value this loop already reads on the next line.
            if (p < n_sink_pages || p >= first_window_page || smem_U[p] == INFINITY) {
                take = true;                               // always resident
            } else if (smem_U[p] > tau) {
                take = true;
            } else if (smem_U[p] == tau && budget > 0) {
                take = true; budget--;
            }
            if (!take) {
                continue;
            }
            // page -> granule union. A granule is loaded if ANY covering page is
            // selected; that is a superset of the selection, never a subset.
            const int t0 = p*page_size;
            const int t1 = t0 + page_size;
            for (int t = t0; t < t1; t += block_tokens) {
                const int blk = (t / block_tokens)*block_tokens;
                if (blk != last_blk && n_emit < stride && (blk / block_tokens) < n_blocks) {
                    L[n_emit++] = blk;
                    last_blk    = blk;
                }
            }
        }
        if (n_emit == 0) {                                  // §7.D: never empty
            L[n_emit++] = 0;
        }
    }
}

// §4.Q's RED CONTROL, and nothing else.
//
// Drops one granule from every list. §4.Q's byte-identity is worthless unless a
// deliberately WRONG selection is shown to break it: if dropping a block leaves
// the output hash unchanged, the list is not reaching the KV loop and every
//
// It drops a MIDDLE entry (n/2), not the last one. Dropping the last was the
// first attempt and it was WRONG: the list is ascending, n_kv is padded to a
// multiple of 256, and the prompt is shorter than the allocation -- so the last
// granule covers cells past the real tokens, which the mask already sets to
// -INF. Removing them cannot move a bit, and the control silently passed while
// proving nothing. A middle granule is inside real, unmasked KV.
// mode 2: keep only the FIRST HALF of the list. Dropping ONE granule turned out
// to be a control too weak to conclude from -- 64 tokens of mid-context natural
// text can genuinely carry ~0 softmax weight, so an unchanged hash is consistent
// BOTH with "the list is ignored" and with "that granule did not matter".
// Halving the list removes the whole recent half of the context and cannot be
// absorbed. Use mode 2 to decide plumbing; mode 1 only to probe sensitivity.
        int * KV_pages_ptr, const int n_lists, const int list_off, const int mode) {
    const int i = blockIdx.x;
    if (i >= n_lists || threadIdx.x != 0) {
        return;
    }
    if (mode == 2) {
        if (*n > 1) {
            *n /= 2;      // ascending list -> drops the recent half outright
        }
        return;
    }
    if (*n > 2) {
        // shift the middle entry out, keeping the list ASCENDING (D4) so the
        // control tests selection content and not ordering.
        int * L = KV_pages_ptr + list_off + (size_t) i*KV_pages_ptr[0];
        const int drop = *n / 2;
        for (int s = drop; s + 1 < *n; ++s) {
            L[s] = L[s + 1];
        }
        *n -= 1;
    }
}

//
// Today this is the plumbing switch only: unset -> false, so the stock kernel
// reaching the backend; the shape guards below stay, because they are kernel
// (§2.2/§4.N gate prefill off) and for one sequence (§2.3 excludes n_seq > 1).
    // loader -- and the backend simply honours what it was handed.
    if (dst->src[5] == nullptr) {
        return false;
    }

    // specified for single-token decode (§2.2/§4.N gate prefill off) and one
    // shaped falls back to the stock kernel, which reads all of KV and is
    // therefore always a safe answer (§7 fail toward attending).
    const ggml_tensor * Q = dst->src[0];
    return cols_per_block == 1 && Q->ne[1] == 1 && Q->ne[3] == 1;
}

// How many pages of this layer's bounds are already FROZEN, keyed by the bounds
// tensor's device pointer.
//
// §4.I completeness is what licenses this: every page below the previous tail is
// complete and, by §4.J.1, not rewritten. So the steady state rebuilds exactly
// the page the last step appended into -- not an O(context) pass, which is what
// §6.5 exists to forbid. That licence is exactly as wide as the append-only
// assumption it rests on, and the decreasing-n_kv branch below is what happens
// when the assumption stops holding.
//
// What this function returns is therefore a FREEZE WATERMARK, not a launch
// origin. The builder is launched over every page and skips the frozen ones
// itself, because §4.O's sentinel can only ever be written BELOW this value and
// has to be reachable in the same launch (ADR-0024, CLEARING).
//
// PLACEMENT IS A KNOWN SMELL: file-static, so it is shared across CUDA contexts
// rather than owned by one. Correct for the single-device single-context case
// this fork runs, and it must move into ggml_backend_cuda_context before any
// multi-context use. Keyed by device pointer, so a freed-and-reallocated cache
// at the same address would look "already built" -- acceptable only because the
// KV cache outlives every graph that reads it.
// llama_kv_cache::get_n_kv() pads to this granularity; see the note below.

    static std::mutex                             mtx;
    static std::unordered_map<const void *, int>  seen;   // bounds ptr -> n_kv at last build

    std::lock_guard<std::mutex> lock(mtx);

    auto it = seen.find(bounds_data);
    const int prev_n_kv = it == seen.end() ? 0 : it->second;
    seen[bounds_data] = n_kv;

    // The page holding the PREVIOUS tail may have been partial, so it is the
    // first page that still needs work. Everything below it is frozen.
    //
    // *** n_kv IS PADDED, SO IT IS NOT A CONTENT WATERMARK. ***
    // llama_kv_cache::get_n_kv() returns GGML_PAD(used_max_p1, 256), so during
    // decode it is CONSTANT for 256 steps and then jumps by 256. Keying the
    // watermark on it directly made page_first == n_pages while it was
    // unchanged, so page_last >= page_first was false and NOTHING was rebuilt
    // for 255 steps out of every 256 -- new keys ranked against records built
    // before they existed. Measured 2026-08-06 at Kne1 23040 -> 23296:
    // five build=1 lines at the crossing, then five build=0, repeating
    //
    // The direction is the unsafe one: a record that does not cover the newest
    // keys UNDER-states the page's bound, so the page ranks too low and can be
    // dropped -- §4.D violated, silently, which is exactly the failure the
    // retrieval gate cannot see. This is the same defect class the window
    // computation already carries a warning about at the head of
    // position is required.
    //
    // The true content length is not host-visible here (the selector reads
    // pos_q out of the mask for precisely this reason), so the fix is
    // conservative rather than exact: back the watermark off by the maximum
    // padding, which is the 256 GGML_PAD granularity. That rebuilds at most
    // ceil(256/page_size) extra pages per step and can never leave the tail
    // stale. Over-rebuilding costs bandwidth; under-rebuilding drops content.
    //
    // *** AND n_kv CAN FALL, WHICH THE BACK-OFF ABOVE DOES NOT COVER. ***
    // about an APPEND-ONLY cache, where prev_n_kv is a floor. It is not one:
    // llama_kv_cache::clear(), a seq_rm deep enough to cross a padding
    // boundary, a prompt-cache checkpoint restore and a context shift all
    // shrink the cache. When that happens page_first is derived from the OLD,
    // larger prev_n_kv, so page_first > page_last = n_pages - 1, nothing below
    // the watermark is rebuilt, and NOTHING is rebuilt at all -- for as long as
    // the cache stays short. New content is then ranked against records
    // accumulated over content that is gone: §4.D violated, silently, in the
    // same unsafe direction as the padding defect.
    //
    // A CLAMP IS NOT THE FIX, and this is the whole design decision. Taking
    // min(prev_n_kv, n_kv) is O(1) and wrong: it freezes every page BELOW the
    // new tail, and a decrease is precisely the event that can have rewritten
    // those rows. build_rope_shift rewrites every cached key in place (§7.J);
    // a seq_rm plus refill lands new keys in freed cells low in the cache
    // (§7.W). The rows a clamp keeps frozen are the rows most likely to be
    // stale. So: any decrease is answered with page_first = 0 -- ONE full
    // rebuild -- which is what §7.X's Recovery clause already rules.
    //
    // WHY §6.5 DOES NOT FORBID THAT, argued rather than asserted, because the
    // next person to touch this function is the one who has to weigh it.
    // §6.5 forbids an O(context) scan ON THE STEADY-STATE PATH. The
    // steady-state path is decode-by-append, and it is monotone: an append
    // cannot lower n_kv, so it cannot reach this branch. Reaching it requires
    // a MUTATION event -- clear, checkpoint restore, context shift, a deep
    // seq_rm -- which is not a step of the steady state but an interruption of
    // it. Two things make that argument load-bearing rather than a definition:
    //   (1) the watermark is still advanced -- the unconditional
    //       `seen[bounds_data] = n_kv` above runs on this path too -- so the
    //       pass is paid ONCE per event and the very next step is tail-only
    //       again. Move that store under an `n_kv >= prev_n_kv` and the full
    //       rebuild WOULD become the steady state, and §6.5 would forbid it
    //       correctly. That is why it is asserted separately in §10's test.
    //   (2) the cost is bounded by a pass this design already pays: rebuilding
    //       whose cost §8 already carries. A mutation event is charged one
    //       extra prefill-shaped bounds pass, not a new order of work.
    // It is LOGGED for the same reason §7.X's Recovery makes logging
    // non-optional: if some caller shrinks the cache every step, this branch
    // plus a scan with nothing announcing it (crosscut.bench-integrity,
    // perf.gates-that-lie). A visible line is what makes that a bug report
    // instead of a slow mystery.
    if (n_kv < prev_n_kv) {
                      "if this repeats every step, that is the bug)\n",
                      __func__, (size_t) bounds_data, prev_n_kv, n_kv,
                      (n_kv + page_size - 1) / page_size);
        return 0;
    }

}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE,
    // arms the producer block below; exactly one of nine launch_fattn call sites passes it
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V);

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(f16_extra.V != 0);
            half * V_f16 = (half *) f16_extra.V;
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // §2.5): both fill `KV_max` with a per-launch skip structure, and exactly
    // one of them may own it. They are mutually exclusive rather than merged
    // because the mask scan only runs for prefill-shaped launches
    // launch wants both -- and a silent merge would be the kind of confounded
    // path that cannot be reasoned about later.
        const int n_head   = Q->ne[2];
        const int nseq     = Q->ne[3];

        //
        // Tail-page-only in the steady state (§6.5). The arithmetic here is the
        // than recomputed at the call site.
        //
        // §4.O CLEARING: page_first is a FREEZE WATERMARK, not the launch
        // origin. The launch covers every page, and the kernel early-outs on a
        // frozen page UNLESS its record carries the invalidation sentinel. It
        // has to be that way round: an invalidated page is by definition below
        // the watermark (a write at or above it is in a page this rebuilds
        // anyway), so a launch that started at page_first could never reach one
        // -- the sentinel would be honoured forever, every invalidated page
        // would attend for the rest of the session, and §8's arithmetic would
        // decay with session churn with nothing announcing it
        // (crosscut.bench-integrity, perf.gates-that-lie). Clearing happens in
        // THIS launch, on the same stream, ahead of the selector below, so not
        // even one selection runs against a sentinel that a rebuild has already
        // been asked for.
        ggml_tensor * bounds    = dst->src[5];
        GGML_ASSERT(page_size > 0);

        const int n_pages    = (K->ne[1] + page_size - 1) / page_size;
        const int page_last  = n_pages - 1;


        // Capacity is n_blocks: the union of ANY page selection over blocks can
        // never exceed the block count of the whole view (§7's fail toward
        // attending applied to granularity).

        KV_max.alloc(lay.total);

        ggml_cuda_kernel_launch_params launch_params =
        // §3.5 step 3 -- the resident set, via the ONE implementation of the
        //
        // THE INVARIANT: the window must cover at least W REAL tokens --
        //
        //
        // and the sink must cover at least n_sink real tokens. Both roundings
        // are outward, TOWARD residency (ceil for the sink, floor for the
        // window), so a partially covered page is resident rather than ranked.
        //
        // *** THE WINDOW START IS NOT COMPUTED HERE, AND MUST NOT BE. ***
        // It needs pos_q, the query's OWN position, which is not a host-visible
        // quantity: K->ne[1] is GGML_PAD(n_tokens, 256) and runs up to 255 cells
        // past the last real token. Feeding that padded length in as pos_q is a
        // shipped defect this comment used to assert was safe ("a superset --
        // fail toward attending"). It is the exact opposite: first_window_page
        // is monotonically INCREASING in pos_q and the window is the page range
        // [first_window_page, n_pages), so an over-estimate RAISES the start and
        // DROPS real tokens off the back of the window -- 941 covered against a
        // declared 1024 at the operating point, which is a silent retrieval
        // miss. The selector reads the true pos_q out of the mask instead.
        //
        // n_sink_pages has no such hazard: ceil(n_sink/B) is a function of the
        // hparams knobs alone and touches neither pos_q nor n_kv, so it stays
        // on the host. It is computed through the same shared header so the two
        // halves of §3.5 cannot drift apart independently.


        // §2.3/§4.N have already restricted this launch to a single sequence and
        // row is row 0 of sequence 0 and needs no stride arithmetic. Assert it
        // rather than assume it: a future relaxation of those guards would
        // otherwise read a wrong row and mis-place the window silently.
        GGML_ASSERT(Q->ne[1] == 1 && Q->ne[3] == 1);


        } else {
            const ggml_tensor * Qt = dst->src[0];
            float scale = 1.0f;
            memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

            // smem: n_pages floats for U_p, plus 64 floats of reduction scratch
            const size_t smem = (n_pages + 64)*sizeof(float);

            ggml_cuda_kernel_launch_params lp_sel =
                lay.n_lists, lay.stride, lay.list_off,
                n_pages, (int) bounds->ne[1], (int) K->ne[1], (int) K->ne[0], (int) K->ne[2],
                n_head, Qt->nb[1], Qt->nb[2], scale);
        }
        CUDA_CHECK(cudaGetLastError());

        if (drop_one) {
            CUDA_CHECK(cudaGetLastError());
        }

        // Read the produced buffer BACK and report on it. Two consumers, ONE
        // readback:
        //
        //     even though the branch demonstrably runs, so the question "did the
        //     drop land in the buffer the kernel reads?" has to be answered by
        //     observation, not by reading the producer source again.
        //     traffic counters. The knob is parsed and range-checked by
        //     llama_hparams (src/llama-hparams.cpp, §6.7); the backend cannot
        //     reach hparams, so it reads the same variable rather than growing a
        //     already rejected anything outside [0,1].
        //
        // Both are off by default and cost nothing when off (§4.S). When ON,
        // is a diagnostic, NOT a timing measurement, and its tok/s must not be
        // quoted as one.

        static bool       dbg_list_done = false;

            const bool want_list     = dbg_list && !dbg_list_done;

            // A readback needs a stream synchronize, and a synchronize is
            // ILLEGAL while the stream is capturing a CUDA graph -- it aborts
            // the process ("operation not permitted when stream is capturing").
            // is_enabled, common.cuh); this check is the belt to that braces,
            // and it REPORTS the skip rather than swallowing it, because a
            // statistics stream with silent holes is worse than none.
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            if (want_readback) {
                CUDA_CHECK(cudaStreamIsCapturing(main_stream, &capture));
            }

            if (want_readback && capture != cudaStreamCaptureStatusNone) {
                    fprintf(stderr,
                        "(the counters need a readback; run with GGML_CUDA_DISABLE_GRAPHS=1)\n",
                        bounds->name);
                    fflush(stderr);
                }
            } else if (want_readback) {
                std::vector<int> h(lay.total);
                CUDA_CHECK(cudaMemcpyAsync(h.data(), KV_max.ptr, lay.total*sizeof(int),
                                           cudaMemcpyDeviceToHost, main_stream));
                CUDA_CHECK(cudaStreamSynchronize(main_stream));

                    // §4.K/§4.S — metadata bytes and KV bytes, SEPARATELY. The
                    // header before changing anything here, in particular why
                    // the two figures are never summed.
                    cfg.n_pages      = n_pages;
                    cfg.n_sink_pages = n_sink_pages;
                    // The host cannot see pos_q (that is the whole point of the
                    // §3.5 fix above), so it bounds it by the padded length. The
                    // window start is monotone in pos_q, so this OVER-states the
                    // ranked range -- the metadata leg errs toward "the scan is
                    // expensive", which is the direction that cannot hide a
                    // metadata-dominated regime.
                    cfg.first_window_page =
                    cfg.n_kv_head      = (int) K->ne[2];
                    cfg.record_bytes   = bounds->nb[1];   // the §6.3 record, from the tensor itself
                    cfg.n_lists        = lay.n_lists;
                    cfg.list_stride    = lay.stride;
                    cfg.list_off       = lay.list_off;
                    cfg.n_head         = n_head;
                    // What the kernel LOADS per token per KV HEAD: one head-dim
                    // row of K plus one of V, in the type it actually reads
                    // (f16 if this launch converted the cache).
                    //
                    // NOT nb11/nb21. Those are the TOKEN strides, and the KV
                    // cache interleaves all four KV heads within a token, so
                    // nb11 is 4x the row this kernel reads (576 B vs 144 B at
                    // q4_0, head_dim 256). Using it inflated kv_bytes by the GQA
                    // KV-head count -- caught by the first real decode, because
                    // the counter's own arithmetic disagreed with §8.2.
                    const size_t k_row = (need_f16_K && K->type != GGML_TYPE_F16)
                        ? K->ne[0]*sizeof(half) : ggml_row_size(K->type, K->ne[0]);
                    const size_t v_row = (need_f16_V && V->type != GGML_TYPE_F16)
                        ? V->ne[0]*sizeof(half) : ggml_row_size(V->type, V->ne[0]);
                    cfg.kv_row_bytes   = k_row + v_row;


                    char line[512];
                    fprintf(stderr, "%s\n", line);
                    fflush(stderr);
                }

                if (want_list) {
                    dbg_list_done = true;
                    // n_pages vs bounds_pages is PROOF OF EXECUTION for the §6.3
                    // record-stride fix, and it is printed rather than derived
                    // because deriving it is exactly what hid the defect: the
                    // selector used n_pages as the per-kv_head record stride
                    // while the builder used bounds->ne[1], and the two are
                    // equal only when the cache is exactly full. Any line where
                    // these two differ is a step on which the old code read
                    // another KV head's bound records.
                    fprintf(stderr,
                        "hdr[C]=%d hdr[O]=%d n_iter[0]=%d n_pages=%d bounds_pages=%d\n",
                        h[0], h[1], n0, n_pages, (int) bounds->ne[1]);
                    const int mid = n0/2;
                    for (int t = (mid-2 < 0 ? 0 : mid-2); t <= mid+2 && t < lay.stride; ++t) {
                        fprintf(stderr, " %d", h[lay.list_off + 0*lay.stride + t]);
                    }
                    fprintf(stderr, "   (drop_one=%d)\n", (int) drop_one);

                    // PROOF OF EXECUTION for the §3.5 window fix, read out of the
                    // buffer the kernel actually produced rather than inferred from
                    // the source. The resident window is the CONTIGUOUS ASCENDING
                    // TAIL of the list (D4 emits in page order), so walking that run
                    // back from the end recovers the window start the DEVICE chose.
                    // Its distance to the last real token is the real-token coverage
                    // the invariant is about: it must be >= the declared W, and the
                    // padded-pos_q defect showed up here as a coverage BELOW W.
                    {
                        const int * L = h.data() + lay.list_off;
                        int run = n0 - 1;
                        const int win_start_tok = n0 > 0 ? L[run] : -1;
                        // The coverage printed is measured against the PADDED view,
                        // because pos_q is the one quantity the host cannot see --
                        // which is the whole reason for this fix. It is therefore an
                        // UPPER BOUND on the real-token coverage, and it is labelled
                        // so: under the padded-pos_q defect this line would have
                        // printed exactly W and looked perfect while the real
                        // coverage was W - 83. Compare `window starts at token`
                        // against first_window_page*page_size instead; that is the
                        // number the device actually chose.
                        fprintf(stderr,
                            "device window starts at token %d (page %d) -> <=%d padded cells "
                            "covered, an UPPER BOUND on real coverage (declared W=%d)\n",
                            win_start_tok, win_start_tok < 0 ? -1 : win_start_tok/page_size,
                            win_start_tok < 0 ? -1 : (int) K->ne[1] - win_start_tok,
                    }
                    fflush(stderr);
                }
            }
        }
    } else
    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
        ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
