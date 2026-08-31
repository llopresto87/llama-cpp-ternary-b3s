#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q2_B3:   return vec_dot_q2_b3_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_B3:   return VDR_Q2_B3_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 5;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
// =====================================================================================
// Bespoke RDNA3 (gfx1100) ternary decode GEMV for q2_0 / q2_b3.
//
// One wave (physical warp) computes exactly one output row for the decode case
// (ncols_dst == 1, single channel/sample, no ids, no fusion). Lanes stride over the
// row's weight blocks; a warp reduction sums the per-lane partials. The int8 dot uses
// ggml_cuda_dp4a (V_DOT4_I32_IU8 / sudot4 on RDNA3). The ternary (c-1) offset is folded
// as -sum(x) via the q8_1 s-term: acc += d * (d8 * sumi - s8).
//
// Decode math is byte-identical to vec_dot_q2_0_q8_1 / vec_dot_q2_b3_q8_1 in vecdotq.cuh;
// only the parallelization (wave-per-row instead of the generic multi-warp block) differs.
// Everything else falls through to the generic mmvq path unchanged.
// =====================================================================================

// nwaves independent warps per block, each computing a DIFFERENT output row
// (row = blockIdx.x*nwaves + threadIdx.y). Larger workgroups pack more waves per
// CU at the small-m per-layer shapes (weights are Infinity-Cache-resident there),
// where 1-wave workgroups under-fill the machine / exhaust workgroup slots. No
// cross-warp cooperation, so no shared memory or barrier is needed. The decode math
// is unchanged and byte-identical to vec_dot_q2_0_q8_1 / vec_dot_q2_b3_q8_1.

// Per-lane partial sums of one output row against NCOLS activation columns (the accumulators
// are the caller's, already zeroed; the caller warp-reduces each one). Lanes stride over the
// evaluate the x-row and the gate-row against the SAME already-resident q8_1 activations.
template <int warp_size>
static __device__ __forceinline__ float ternary_row_partial_q2_0(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int row, const int nblocks, const int stride_row_x, const int lane) {
    const block_q2_0 * xrow = (const block_q2_0 *) vx + (size_t) row * stride_row_x;

    // NOTE: q2_0 keeps BLOCK-granular lane striding, unlike q2_b3. Sub-block striding was
    // measured on gfx1100 (tg128, Q2_g64): 66.54 -> 65.34 t/s, a regression. q2_0 runs
    // nwaves=2 and its 64-weight block amortizes the `d` load over two sub-blocks, so the
    // per-unit reload costs more than the residual imbalance saves. Per-type, like nwaves.
    float acc = 0.0f;
    for (int ib = lane; ib < nblocks; ib += warp_size) {
        const block_q2_0 * bq = xrow + ib;
        const float d = (float) bq->d;
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {                              // two 32-wide sub-blocks
            const block_q8_1 * bq8 = y + ib*2 + sub;                     // qk/QK8_1 == 2
            const float d8 = __low2float(bq8->ds);
            const float s8 = __high2float(bq8->ds);                      // d8 * sum(x_quant) = sum(x_real)
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {                                // 8 code bytes = 32 2-bit codes
                const uint8_t byte = bq->qs[sub*8 + j];
                const int codes = ( byte        & 0x03)
                                | ((int)((byte>>2)&0x03) << 8)
                                | ((int)((byte>>4)&0x03) << 16)
                                | ((int)((byte>>6)&0x03) << 24);
                const int u = get_int_b4(bq8->qs, j);
                sumi = ggml_cuda_dp4a(codes, u, sumi);
            }
            acc += d * (d8 * sumi - s8);
        }
    }
    return acc;
}

template <int warp_size>
static __device__ __forceinline__ float ternary_row_partial_q2_b3(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int row, const int nblocks, const int stride_row_x, const int lane) {
    const block_q2_b3 * xrow = (const block_q2_b3 *) vx + (size_t) row * stride_row_x;

    // Stride lanes over 32-weight CHUNKS, not whole 128-weight blocks. At the decode shapes
    // that dominate this model (k=5120 => 40 blocks/row) block-granular striding leaves
    // 40/32 = 1.25 blocks per lane, i.e. a 2x load imbalance across the wave with the warp
    // reduction paid regardless. Chunk granularity gives 160/32 = exactly 5 units per lane.
    const int nunits = nblocks * 4;
    float acc = 0.0f;
    for (int u = lane; u < nunits; u += warp_size) {
        const int ib    = u >> 2;
        const int chunk = u & 3;
        const block_q2_b3 * bq = xrow + ib;
        const uint8_t * qs = bq->qs;
        {
            const float d  = (float) bq->d;                              // one scale per 128
            const block_q8_1 * bq8 = y + ib*4 + chunk;                   // qk/QK8_1 == 4
            const float d8 = __low2float(bq8->ds);
            const float s8 = __high2float(bq8->ds);

            // Cheap base-3 unpack: one pre-expanded LUT load per dense byte gives four
            // int8 code-lanes (bits 0/8/16/24) plus the fifth trit parked at bits 30..31.
            // The eight dp4a code-words are then assembled with shift/mask/or only; no
            // per-element LUT gather. Straggler trits (elements 30,31 of the chunk) come
            // from the shared byte 24/25 via the original 5-trit LUT. Bit-identical to the
            // per-element decode, guarded exhaustively over all 256 byte values by
            // scripts/tests/test_b3lut_unpack.py.
            const uint8_t * cqs = qs + 6*chunk;
            const uint32_t T0 = ggml_cuda_b3lut_x[cqs[0]];
            const uint32_t T1 = ggml_cuda_b3lut_x[cqs[1]];
            const uint32_t T2 = ggml_cuda_b3lut_x[cqs[2]];
            const uint32_t T3 = ggml_cuda_b3lut_x[cqs[3]];
            const uint32_t T4 = ggml_cuda_b3lut_x[cqs[4]];
            const uint32_t T5 = ggml_cuda_b3lut_x[cqs[5]];

            const uint16_t Lstr = ggml_cuda_b3lut[qs[24 + (chunk >> 1)]];
            const int      sd   = 2 * (chunk & 1);
            const uint32_t str0 = (Lstr >> (2 * sd))     & 0x3u;
            const uint32_t str1 = (Lstr >> (2 * sd + 2)) & 0x3u;

            const uint32_t c0 = T0 & 0x03030303u;
            const uint32_t c1 = (T0 >> 30)                 | (T1 << 8);
            const uint32_t c2 = ((T1 >> 24) & 0x3u)        | ((T1 >> 30) << 8) | (T2 << 16);
            const uint32_t c3 = ((T2 >> 16) & 0x0303u)     | ((T2 >> 30) << 16) | (T3 << 24);
            const uint32_t c4 = ((T3 >> 8)  & 0x030303u)   | ((T3 >> 30) << 24);
            const uint32_t c5 = T4 & 0x03030303u;
            const uint32_t c6 = (T4 >> 30)                 | (T5 << 8);
            const uint32_t c7 = ((T5 >> 24) & 0x3u)        | ((T5 >> 30) << 8) | (str0 << 16) | (str1 << 24);

            // Two independent dp4a accumulator chains so RDNA3 can VOPD-dual-issue the
            // unpack ALU against the sudot4 dots (int add is exact/associative).
            int s0 = 0, s1 = 0;
            s0 = ggml_cuda_dp4a((int) c0, get_int_b4(bq8->qs, 0), s0);
            s1 = ggml_cuda_dp4a((int) c1, get_int_b4(bq8->qs, 1), s1);
            s0 = ggml_cuda_dp4a((int) c2, get_int_b4(bq8->qs, 2), s0);
            s1 = ggml_cuda_dp4a((int) c3, get_int_b4(bq8->qs, 3), s1);
            s0 = ggml_cuda_dp4a((int) c4, get_int_b4(bq8->qs, 4), s0);
            s1 = ggml_cuda_dp4a((int) c5, get_int_b4(bq8->qs, 5), s1);
            s0 = ggml_cuda_dp4a((int) c6, get_int_b4(bq8->qs, 6), s0);
            s1 = ggml_cuda_dp4a((int) c7, get_int_b4(bq8->qs, 7), s1);

            acc += d * (d8 * (float)(s0 + s1) - s8);
        }
    }
    return acc;
}

// One wave computes one output row. With has_gate the SAME wave also computes the gate
// row for that output, so the q8_1 activation blocks are fetched once and dotted twice --
// this is why the fused FFN gate/up projection belongs on this path rather than falling
// through to the generic mmvq kernel.
//
// NCOLS is the activation batch width. NCOLS == 1 is the shipped decode path. For NCOLS > 1
// each wave walks the same weight row once per column, so the row's decode is reused across
// columns; the activation base advances by stride_col_y block_q8_1 units per column and the
// destination by stride_col_dst floats. Gate/bias fusion is excluded at NCOLS > 1 (enforced
// at the dispatch gate), so has_gate is false in those instantiations.
template <int warp_size, int nwaves, bool is_b3, bool has_gate, int NCOLS>
static __global__ void __launch_bounds__(warp_size*nwaves, 1) mul_mat_vec_ternary_decode(
        const void * __restrict__ vx, const void * __restrict__ vgate, const void * __restrict__ vy,
        float * __restrict__ dst, const float * __restrict__ x_bias, const float * __restrict__ gate_bias,
        const int ncols_x, const int stride_row_x, const int nrows_x, const ggml_glu_op glu_op,
        const int stride_col_y, const int stride_col_dst) {
    const int row = blockIdx.x*nwaves + threadIdx.y;
    if (row >= nrows_x) {
        return;
    }
    const int lane    = threadIdx.x;
    const int nblocks = ncols_x / (is_b3 ? QK2_B3 : QK2_0);
    const block_q8_1 * y = (const block_q8_1 *) vy;

    static_assert(NCOLS == 1 || !has_gate, "batch-N excludes gate fusion");

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        const block_q8_1 * yj = y + (size_t) j * stride_col_y;

        float acc;
        if constexpr (is_b3) {
            acc = ternary_row_partial_q2_b3<warp_size>(vx, yj, row, nblocks, stride_row_x, lane);
        } else {
            acc = ternary_row_partial_q2_0<warp_size>(vx, yj, row, nblocks, stride_row_x, lane);
        }
        acc = warp_reduce_sum<warp_size>(acc);

        float gate_acc = 0.0f;
        if constexpr (has_gate) {
            if constexpr (is_b3) {
                gate_acc = ternary_row_partial_q2_b3<warp_size>(vgate, yj, row, nblocks, stride_row_x, lane);
            } else {
                gate_acc = ternary_row_partial_q2_0<warp_size>(vgate, yj, row, nblocks, stride_row_x, lane);
            }
            gate_acc = warp_reduce_sum<warp_size>(gate_acc);
        }

        if (lane != 0) {
            continue;
        }

        // Epilogue mirrors the generic mmvq fusion epilogue exactly (x_scale/gate_scale are
        // NVFP4-only and cannot be set for a ternary type, so they are absent here).
        // x_bias and gate_bias are independently optional, exactly as in the generic path
        // (which zero-initializes the bias arrays); only lane 0 gets here, so the branches
        // cost nothing measurable.
        float result = acc;
        if (x_bias != nullptr) {
            result += x_bias[row];
        }
        if constexpr (has_gate) {
            float gate_value = gate_acc;
            if (gate_bias != nullptr) {
                gate_value += gate_bias[row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                    break;
                default:
                    result = result * gate_value;
                    break;
            }
        }
        dst[(size_t) j * stride_col_dst + row] = result;
    }
}

// Paired ternary GEMV: two INDEPENDENT projections that share the same activation vector,
// issued as one dispatch over the concatenation of their rows.
//
// Rationale (measured, gfx1100). Achieved bandwidth scales with how much a single dispatch
// moves: 42 MB shapes reach 750 GB/s and the 20 MB lm_head 940 GB/s, but the 7.4 MB
// attn_gate stalls at 372 GB/s and 12 MB attn_qkv at 589 GB/s -- small kernels cannot
// amortize their launch ramp. That deficit is NOT a parallelism problem: both re-
// decompositions were measured strictly worse (multi-row R=2/4/8 -> 63.78/52.64/37.83;
// split-k NSPLIT=2/4/8 -> 67.00/66.11/65.85, vs 68.80 one-wave-per-row). The only way up is
// to make each dispatch bigger.
//
// qkv (m=10240) and gate (m=6144) are independent, share `input`, and occur in all 48
// linear-attention layers: 20.87 + 19.82 = 40.69 us today for 19.66 MB (483 GB/s). As one
// m=16384 dispatch in the 750 GB/s regime that is ~26 us.
//
// Rows below nrows_a come from the first matrix, the rest from the second. With one wave
// per row there is no intra-wave divergence at the boundary.
// Up to this many independent projections may share one dispatch.
#define TQ_MULTI_MAX TQ_CUDA_MULTI_MAX

struct tq_multi_args {
    const void * vx[TQ_MULTI_MAX];
    float *      dst[TQ_MULTI_MAX];
    int          row_end[TQ_MULTI_MAX];        // cumulative row boundaries; row_end[N-1] == total
    int          stride_col_dst[TQ_MULTI_MAX]; // PER-SLOT dst column stride (that member's rows)
};

template <int warp_size, int nwaves, bool is_b3, int N, int NCOLS>
static __global__ void __launch_bounds__(warp_size*nwaves, 1) mul_mat_vec_ternary_decode_multi(
        const tq_multi_args args, const void * __restrict__ vy,
        const int ncols_x, const int stride_row_x, const int stride_col_y) {
    // The verify-width instantiations keep the batch-N geometry of one wave per row (see
    // mul_mat_vec_q_ternary_decode_nwaves for why); the launch site never combines them.
    static_assert(NCOLS == 1 || nwaves == 1, "N-way at NCOLS > 1 is one wave per row");
    const int row = blockIdx.x*nwaves + threadIdx.y;
    if (row >= args.row_end[N-1]) {
        return;
    }
    const int lane    = threadIdx.x;
    const int nblocks = ncols_x / (is_b3 ? QK2_B3 : QK2_0);
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Select the owning matrix with COMPILE-TIME indices. Indexing args.vx[] by a runtime
    // value would push the whole struct into scratch memory; measured, that cost the fused
    // 4-way shape 634 GB/s against 812 GB/s for the equivalent select-based pair kernel.
    // The unrolled loop keeps every subscript constant, so these stay in registers -- the
    // per-slot stride_col_dst is resolved in the SAME loop for the same reason.
    const void * vx   = args.vx[0];
    float *      dstp = args.dst[0];
    int          base = 0;
    int          stride_col_dst = args.stride_col_dst[0];
#pragma unroll
    for (int t = 1; t < N; ++t) {
        if (row >= args.row_end[t-1]) {
            vx   = args.vx[t];
            dstp = args.dst[t];
            base = args.row_end[t-1];
            stride_col_dst = args.stride_col_dst[t];
        }
    }
    const int r = row - base;

    // One accumulator per activation column, mirroring the batch-N decode kernel. The shared
    // activation's columns sit at j*stride_col_y in the q8_1 buffer (the main path's layout,
    // member's dst columns sit at j*stride_col_dst for ITS OWN row count.
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        const block_q8_1 * yj = y + (size_t) j * stride_col_y;

        float acc;
        if constexpr (is_b3) {
            acc = ternary_row_partial_q2_b3<warp_size>(vx, yj, r, nblocks, stride_row_x, lane);
        } else {
            acc = ternary_row_partial_q2_0<warp_size>(vx, yj, r, nblocks, stride_row_x, lane);
        }
        acc = warp_reduce_sum<warp_size>(acc);

        if (lane == 0) {
            dstp[(size_t) j * stride_col_dst + r] = acc;
        }
    }
}

// Waves-per-block is PER-TYPE (context-adaptive dispatch). Measured on gfx1100
// (RX 7900 XTX), tg128 over the Ternary-Bonsai-27B decode workload:
//   q2_0 : nwaves=2 is best (60.2 -> 61.6); nwaves=1/4/8 are flat-to-worse.
//   q2_b3: nwaves=1 is best; nwaves>=2 regresses (heavier LUT-decode kernel has
//          higher VGPR pressure, so larger blocks cut occupancy).
// Each block runs `nwaves` independent warps on distinct rows; grid = ceil(nrows/nwaves).
#define TQ_TERNARY_NWAVES_Q2_0  2
#define TQ_TERNARY_NWAVES_Q2_B3 1

// Sweep/override knobs. These exist so the tuned constants above can be re-measured from a
// SINGLE binary (no rebuild, no stash) whenever the kernel body changes and invalidates an
// earlier sweep -- which is exactly what happened to the nwaves numbers above.
//   TQ_TERNARY_NWAVES_B3 / _Q20 : override waves-per-block (1, 2, 4 or 8).
//   TQ_HIP_NO_TERNARY_FUSION    : force gate/bias-fused GEMV back to the generic mmvq path,
//                                 so old vs new can be A/B'd for output equality in place.
static int tq_ternary_env(const char * name, int fallback) {
    const char * s = getenv(name);
    if (s == nullptr) {
        return fallback;
    }
    const int v = atoi(s);
    return (v == 1 || v == 2 || v == 4 || v == 8) ? v : fallback;
}

static bool tq_ternary_fusion_disabled() {
    static const bool disabled = getenv("TQ_HIP_NO_TERNARY_FUSION") != nullptr;
    return disabled;
}

// TQ_HIP_NO_Q8_1_CACHE=1 restores one src1 quantization per matmul, for A/B from one binary.
static bool tq_no_q8_1_cache() {
    static const bool disabled = getenv("TQ_HIP_NO_Q8_1_CACHE") != nullptr;
    return disabled;
}

// Widest activation the q8_1 cache will hold. 1 is plain decode; 2 is the speculative verify
// step at the shipped n_max = 1, which is itself derived from the fattn VEC width bound
// ggml-alloc keeps a node's contents alive from producer to last consumer whatever its column
// count -- so this bound is a retained-pool-memory budget, not a correctness limit: raising it
// costs ~5.5 KB per distinct activation per extra column. It must stay a `<=` bound; pinning it
// to a single width would take the cache away from the width it already serves.
#define TQ_Q8_1_CACHE_MAX_NE11 2

// TQ_HIP_NO_TERNARY_BATCH_N=1 sends ncols_dst > 1 back to the generic mmvq kernel, so the
// batch-N body can be A/B'd against the incumbent from ONE binary. ncols_dst == 1 is the
// shipped decode path and is unaffected by this variable in either state; the arm that ran is
// witnessed by the kernel name in a dispatch trace, not inferred from the timing.
static bool tq_ternary_batch_n_disabled() {
    static const bool disabled = getenv("TQ_HIP_NO_TERNARY_BATCH_N") != nullptr;
    return disabled;
}

// TQ_HIP_NO_NWAY_W2=1 narrows the N-way pair admission back to batch-1 -- the exact pre-W5
// dispatch pattern -- so the verify-width (ne11 == 2) consolidation can be A/B'd from ONE
// binary. Read ONCE into a static: an env flipped mid-run must not change behaviour between
// paired arms (same discipline as TQ_HIP_NO_TERNARY_BATCH_N above).
static bool tq_nway_w2_disabled() {
    static const bool disabled = getenv("TQ_HIP_NO_NWAY_W2") != nullptr;
    return disabled;
}

template <int warp_size, int nwaves, bool is_b3, int NCOLS>
static void mul_mat_vec_q_ternary_decode_launch(
        const void * vx, const void * vgate, const void * vy, float * dst,
        const float * x_bias, const float * gate_bias, const ggml_glu_op glu_op,
        const int ncols_x, const int nrows_x, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    const dim3 block_nums((nrows_x + nwaves - 1) / nwaves, 1, 1);
    const dim3 block_dims(warp_size, nwaves, 1);
    if constexpr (NCOLS == 1) {
        if (vgate != nullptr) {
            mul_mat_vec_ternary_decode<warp_size, nwaves, is_b3, true, 1><<<block_nums, block_dims, 0, stream>>>(
                vx, vgate, vy, dst, x_bias, gate_bias, ncols_x, stride_row_x, nrows_x, glu_op, stride_col_y, stride_col_dst);
        } else {
            mul_mat_vec_ternary_decode<warp_size, nwaves, is_b3, false, 1><<<block_nums, block_dims, 0, stream>>>(
                vx, vgate, vy, dst, x_bias, gate_bias, ncols_x, stride_row_x, nrows_x, glu_op, stride_col_y, stride_col_dst);
        }
    } else {
        // Batch-N excludes gate/bias fusion: has_gate is false in the instantiation and the
        // gate/bias pointers are literal nulls at the launch site, so no fused argument can
        // reach the kernel even if the dispatch gate were ever relaxed by mistake.
        GGML_ASSERT(vgate == nullptr && x_bias == nullptr && gate_bias == nullptr);
        mul_mat_vec_ternary_decode<warp_size, nwaves, is_b3, false, NCOLS><<<block_nums, block_dims, 0, stream>>>(
            vx, nullptr, vy, dst, nullptr, nullptr, ncols_x, stride_row_x, nrows_x, glu_op, stride_col_y, stride_col_dst);
    }
}

// Runtime nwaves selection. Templated on nwaves for codegen, chosen per call.
//
// Waves-per-block is a batch-1 knob. The per-type constants above were measured with one
// accumulator and one dp4a chain per wave; a batch-N wave carries NCOLS of each, and the generic
// kernel it displaces already runs one wave per row at ncols_dst > 1 (calc_nwarps and
// calc_rows_per_block both return 1 there). Batch-N therefore keeps exactly that geometry --
// block (warp_size,1,1), grid (nrows_x,1,1) -- which is what makes the A/B a pure kernel-body
// swap, and it is why nwaves > 1 is never instantiated for NCOLS > 1.
template <int warp_size, bool is_b3, int NCOLS>
static void mul_mat_vec_q_ternary_decode_nwaves(
        const int nwaves,
        const void * vx, const void * vgate, const void * vy, float * dst,
        const float * x_bias, const float * gate_bias, const ggml_glu_op glu_op,
        const int ncols_x, const int nrows_x, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    if constexpr (NCOLS == 1) {
        switch (nwaves) {
            case 2:
                mul_mat_vec_q_ternary_decode_launch<warp_size, 2, is_b3, NCOLS>(vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
                break;
            case 4:
                mul_mat_vec_q_ternary_decode_launch<warp_size, 4, is_b3, NCOLS>(vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
                break;
            case 8:
                mul_mat_vec_q_ternary_decode_launch<warp_size, 8, is_b3, NCOLS>(vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
                break;
            default:
                mul_mat_vec_q_ternary_decode_launch<warp_size, 1, is_b3, NCOLS>(vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
                break;
        }
    } else {
        GGML_ASSERT(nwaves == 1);
        mul_mat_vec_q_ternary_decode_launch<warp_size, 1, is_b3, NCOLS>(vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
    }
}

// Widest activation batch the bespoke path serves. Bounded by MMVQ_MAX_BATCH_SIZE: above it
// ggml_cuda_should_use_mmvq sends the matmul to MMQ and this kernel is never reached at all.
#define TQ_TERNARY_MAX_NCOLS 8
static_assert(TQ_TERNARY_MAX_NCOLS <= MMVQ_MAX_BATCH_SIZE,
              "the ternary decode path cannot claim widths MMVQ never routes to it");

// Runtime ncols_dst -> compile-time NCOLS. One arm per admissible width, every template
// argument a literal: a runtime value must never reach a fixed-NCOLS instantiation, which is
// the silent launch-geometry/template mismatch this file has already been burned by once. The
// default arm aborts loudly rather than falling through to some instantiation -- the caller's
// dispatch gate is what bounds ncols_dst to [1, TQ_TERNARY_MAX_NCOLS].
template <int warp_size, bool is_b3>
static void mul_mat_vec_q_ternary_decode_ncols(
        const int ncols_dst, const int nwaves,
        const void * vx, const void * vgate, const void * vy, float * dst,
        const float * x_bias, const float * gate_bias, const ggml_glu_op glu_op,
        const int ncols_x, const int nrows_x, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    static_assert(TQ_TERNARY_MAX_NCOLS == 8, "one switch arm per admissible NCOLS");
#define TQ_TERNARY_NCOLS_CASE(N)                                                                  \
    case N:                                                                                         \
        mul_mat_vec_q_ternary_decode_nwaves<warp_size, is_b3, N>(                                   \
            nwaves, vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x,  \
            stride_col_y, stride_col_dst, stream);                                                  \
        break;
    switch (ncols_dst) {
        TQ_TERNARY_NCOLS_CASE(1)
        TQ_TERNARY_NCOLS_CASE(2)
        TQ_TERNARY_NCOLS_CASE(3)
        TQ_TERNARY_NCOLS_CASE(4)
        TQ_TERNARY_NCOLS_CASE(5)
        TQ_TERNARY_NCOLS_CASE(6)
        TQ_TERNARY_NCOLS_CASE(7)
        TQ_TERNARY_NCOLS_CASE(8)
        default:
            GGML_ABORT("ternary decode: unsupported ncols_dst %d", ncols_dst);
    }
#undef TQ_TERNARY_NCOLS_CASE
}

// Launch the bespoke ternary decode kernel. Caller guarantees the decode preconditions.
static void mul_mat_vec_q_ternary_decode(
        const void * vx, const ggml_type type_x, const void * vgate, const void * vy, float * dst,
        const float * x_bias, const float * gate_bias, const ggml_glu_op glu_op,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    // Waves-per-block is a batch-1 knob (and so is its env override); batch-N is one wave per
    // row, see mul_mat_vec_q_ternary_decode_nwaves.
    if (type_x == GGML_TYPE_Q2_0) {
        static const int nwaves_tuned = tq_ternary_env("TQ_TERNARY_NWAVES_Q20", TQ_TERNARY_NWAVES_Q2_0);
        const int nwaves = ncols_dst == 1 ? nwaves_tuned : 1;
        if (warp_size == 64) {
            mul_mat_vec_q_ternary_decode_ncols<64, false>(ncols_dst, nwaves, vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
        } else {
            mul_mat_vec_q_ternary_decode_ncols<32, false>(ncols_dst, nwaves, vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
        }
    } else { // GGML_TYPE_Q2_B3
        static const int nwaves_tuned = tq_ternary_env("TQ_TERNARY_NWAVES_B3", TQ_TERNARY_NWAVES_Q2_B3);
        const int nwaves = ncols_dst == 1 ? nwaves_tuned : 1;
        if (warp_size == 64) {
            mul_mat_vec_q_ternary_decode_ncols<64, true>(ncols_dst, nwaves, vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
        } else {
            mul_mat_vec_q_ternary_decode_ncols<32, true>(ncols_dst, nwaves, vx, vgate, vy, dst, x_bias, gate_bias, glu_op, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
        }
    }
}

// Host entry for the paired ternary GEMV. Preconditions are checked by
// ggml_cuda_should_fuse_mul_mat_vec_q_pair; this only dispatches.
void ggml_cuda_mul_mat_vec_q_multi(
        ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n_nodes) {
    GGML_ASSERT(n_nodes >= 2 && n_nodes <= TQ_MULTI_MAX);
    const ggml_tensor * src0_a = nodes[0]->src[0];
    const ggml_tensor * src1   = nodes[0]->src[1];
    cudaStream_t stream = ctx.stream();

    const int64_t ne10        = src1->ne[0];
    const int64_t ne11        = src1->ne[1];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    // The pair predicate admits at most TQ_Q8_1_CACHE_MAX_NE11 activation columns, and the
    // kernel below is instantiated for NCOLS in {1, 2} only: raising the bound needs new
    // instantiation arms first, and both assertions below keep that raise LOUD.
    static_assert(TQ_Q8_1_CACHE_MAX_NE11 == 2,
                  "mul_mat_vec_ternary_decode_multi instantiates NCOLS in {1, 2} only");
    GGML_ASSERT(ne11 >= 1 && ne11 <= TQ_Q8_1_CACHE_MAX_NE11);

    // Sized by ne11 exactly like the main path (ne12 == ne13 == 1 is guaranteed by the pair
    // predicate). The entry this seeds carries the main path's layout -- column j at
    // j*(ne10_padded/QK8_1) -- so a same-width main-path consumer may reuse it in place
    const size_t  q8_1_nelem  = ne11 * ne10_padded * sizeof(block_q8_1)/QK8_1;

    // All group members consume the same src1 node, so this hits the graph-scoped q8_1 cache
    // whenever anything else in the layer already quantized it, and seeds it otherwise.
    // The lookup keys on ne11 and the entry records the REAL ne11, so a width-1 entry seeded
    // elsewhere is a MISS for a width-2 consumer (and vice versa) instead of an out-of-bounds
    // read; a miss falls through to a fresh quantization and never evicts, because a
    // narrower consumer may still be pending on the entry it missed.
    char * src1_q8_1_ptr = nullptr;
    for (const auto & e : ctx.q8_1_cache) {
        if (e.src1 == src1 && e.src0_type == src0_a->type && e.ne10 == ne10
                && e.ne11 == ne11 && e.alloc_nelem >= q8_1_nelem) {
            src1_q8_1_ptr = e.buf;
            break;
        }
    }
    if (src1_q8_1_ptr == nullptr) {
        ctx.q8_1_cache_allocs.emplace_back(new ggml_cuda_pool_alloc<char>(ctx.pool(), q8_1_nelem));
        src1_q8_1_ptr = ctx.q8_1_cache_allocs.back()->get();
        ctx.q8_1_cache.push_back({ src1, src0_a->type, ne10, ne11, q8_1_nelem, src1_q8_1_ptr });
        const int64_t s11 = src1->nb[1] / ggml_type_size(src1->type);
        const int64_t s12 = src1->nb[2] / ggml_type_size(src1->type);
        const int64_t s13 = src1->nb[3] / ggml_type_size(src1->type);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1_ptr, src0_a->type,
                               ne10, s11, s12, s13, ne10_padded, ne11, 1, 1, stream);
    }

    tq_multi_args args{};
    int cum = 0;
    for (int t = 0; t < n_nodes; ++t) {
        cum += (int) nodes[t]->src[0]->ne[1];
        args.vx[t]             = nodes[t]->src[0]->data;
        args.dst[t]            = (float *) nodes[t]->data;
        args.row_end[t]        = cum;
        args.stride_col_dst[t] = (int) nodes[t]->ne[0]; // contiguous dst: column j at j*ne0
    }
    // Unused slots repeat the last boundary so the compile-time unrolled search is safe.
    for (int t = n_nodes; t < TQ_MULTI_MAX; ++t) {
        args.vx[t]             = args.vx[n_nodes-1];
        args.dst[t]            = args.dst[n_nodes-1];
        args.row_end[t]        = cum;
        args.stride_col_dst[t] = args.stride_col_dst[n_nodes-1];
    }

    const int nrows_total = cum;
    const int stride_row  = src0_a->nb[1] / ggml_type_size(src0_a->type);
    const int warp_size   = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const bool is_b3      = src0_a->type == GGML_TYPE_Q2_B3;

    // Waves-per-block for the FUSED shapes is tuned separately from the single-matrix
    // kernel. Achieved bandwidth rises with dispatch size (7.4 MB -> 372 GB/s, 19.8 MB ->
    // 648, 41.8 MB -> 738, 357 MB -> 919), and fusion moves these shapes into the larger
    // regime where the single-matrix tuning (nwaves=1, chosen against small shapes) may no
    // longer be right. TQ_TERNARY_MULTI_NWAVES overrides for sweeping.
    const int nwaves_default = is_b3 ? TQ_TERNARY_NWAVES_Q2_B3 : TQ_TERNARY_NWAVES_Q2_0;
    static const int nwaves_tuned = tq_ternary_env("TQ_TERNARY_MULTI_NWAVES", nwaves_default);
    // Waves-per-block is a batch-1 knob here as everywhere (see
    // mul_mat_vec_q_ternary_decode_nwaves): at the verify width the group keeps the batch-N
    // geometry of one wave per row, and NCOLS > 1 is never instantiated with nwaves > 1.
    const int nwaves = ne11 == 1 ? nwaves_tuned : 1;

    const int stride_col_y = (int) (ne10_padded / QK8_1);

    const dim3 block_nums((nrows_total + nwaves - 1) / nwaves, 1, 1);
    const dim3 block_dims(warp_size, nwaves, 1);

#define TQ_LAUNCH_MULTI(WS, NW, B3, N, NC)                                                       \
    mul_mat_vec_ternary_decode_multi<WS, NW, B3, N, NC><<<block_nums, block_dims, 0, stream>>>(   \
        args, src1_q8_1_ptr, ne10, stride_row, stride_col_y)

#define TQ_LAUNCH_MULTI_N(WS, NW, B3, NC)                       \
    switch (n_nodes) {                                          \
        case 2: TQ_LAUNCH_MULTI(WS, NW, B3, 2, NC); break;      \
        case 3: TQ_LAUNCH_MULTI(WS, NW, B3, 3, NC); break;      \
        default: TQ_LAUNCH_MULTI(WS, NW, B3, 4, NC); break;     \
    }

// The template's wave count MUST match the launch geometry: block_dims/block_nums are
// computed from the runtime `nwaves`, so dispatch on it rather than baking in the
// compile-time default (doing the latter launched N waves into a kernel compiled for one).
// The ne11 == 1 arm is the exact pre-W5 dispatch; NCOLS = 2 exists only with one wave per
// row, so the NCOLS > 1 x nwaves > 1 combination is never even instantiated.
#define TQ_LAUNCH_MULTI_W(WS, B3)                               \
    if (ne11 == 1) {                                            \
        switch (nwaves) {                                       \
            case 2:  TQ_LAUNCH_MULTI_N(WS, 2, B3, 1); break;    \
            case 4:  TQ_LAUNCH_MULTI_N(WS, 4, B3, 1); break;    \
            default: TQ_LAUNCH_MULTI_N(WS, 1, B3, 1); break;    \
        }                                                       \
    } else {                                                    \
        TQ_LAUNCH_MULTI_N(WS, 1, B3, 2);                        \
    }

    if (warp_size == 64) {
        if (is_b3) { TQ_LAUNCH_MULTI_W(64, true); }
        else       { TQ_LAUNCH_MULTI_W(64, false); }
    } else {
        if (is_b3) { TQ_LAUNCH_MULTI_W(32, true); }
        else       { TQ_LAUNCH_MULTI_W(32, false); }
    }
#undef TQ_LAUNCH_MULTI_W
#undef TQ_LAUNCH_MULTI_N
#undef TQ_LAUNCH_MULTI
}

// Two adjacent MUL_MAT nodes qualify for the paired ternary GEMV when they are the same
// ternary type, read the SAME src1 node, agree on k and row stride, and are a GEMV of at
// most TQ_Q8_1_CACHE_MAX_NE11 activation columns with no ids and no fusion epilogue.
bool ggml_cuda_should_fuse_mul_mat_vec_q_pair(const ggml_tensor * a, const ggml_tensor * b) {
    // same reason as the fast-path gate in mul_mat_vec_q_switch_type: this fusion ends in the
    // RDNA3 ternary kernel, and q2_0 is an upstream type that must keep its own path elsewhere
    if (!GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (a == nullptr || b == nullptr) {
        return false;
    }
    const ggml_tensor * a0 = a->src[0];
    const ggml_tensor * b0 = b->src[0];
    if (a0 == nullptr || b0 == nullptr) {
        return false;
    }
    if (a0->type != b0->type) {
        return false;
    }
    if (a0->type != GGML_TYPE_Q2_0 && a0->type != GGML_TYPE_Q2_B3) {
        return false;
    }
    if (a->src[1] != b->src[1] || a->src[1] == nullptr) {           // same activation node
        return false;
    }
    if (a->src[2] != nullptr || b->src[2] != nullptr) {             // no MUL_MAT_ID
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    if (a->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (a0->ne[0] != b0->ne[0]) {                                   // same k
        return false;
    }
    if (a0->nb[1] != b0->nb[1]) {                                   // same row stride
        return false;
    }
    // WIDTH GATE: the shared activation may carry at most TQ_Q8_1_CACHE_MAX_NE11 columns --
    // the q8_1 cache bound -- and TQ_HIP_NO_NWAY_W2=1 narrows admission back to batch-1 (the
    // exact pre-W5 dispatch pattern) so the widening can be A/B'd from ONE binary.
    //
    // TWO reasons bound this width, and satisfying only the first arms a silent bug
    // (1) Kernel geometry: mul_mat_vec_ternary_decode_multi carries per-column accumulators
    //     for NCOLS in {1, 2} ONLY, one wave per row at NCOLS > 1.
    // (2) q8_1 CACHE SIZING: ggml_cuda_mul_mat_vec_q_multi writes ctx.q8_1_cache under the
    //     same key as the main path. The main path serves up to TQ_Q8_1_CACHE_MAX_NE11
    //     columns and reads column j at j*(ne10_padded/QK8_1); an entry seeded NARROWER than
    //     its consumer is an out-of-bounds device read with no fault -- garbage in the
    //     speculative verify step's second column of logits. The N-way path therefore sizes
    //     its allocation by ne11 and keys its lookup on ne11 (a narrow entry is a MISS for a
    //     wider consumer, never a reuse), and this gate keeps any activation wider than the
    //     cache bound out of the N-way path entirely, so no admitted width can ever outrun
    //     what the N-way path seeds. Whoever widens this path further must move the kernel
    //     instantiations, the allocation, the lookup key and this bound TOGETHER.
    const ggml_tensor * s1 = a->src[1];
    const int64_t max_ne11 = tq_nway_w2_disabled() ? 1 : TQ_Q8_1_CACHE_MAX_NE11;
    if (s1->ne[1] < 1 || s1->ne[1] > max_ne11 || s1->ne[2] != 1 || s1->ne[3] != 1) {
        return false;
    }
    if (a0->ne[2] != 1 || a0->ne[3] != 1 || b0->ne[2] != 1 || b0->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return false;
    }
    if (!ggml_is_contiguous(s1)) {
        return false;
    }
    return true;
}

static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    // Bespoke RDNA3 ternary decode fast path: GEMV over q2_0 / q2_b3 for an activation batch of
    // 1..TQ_TERNARY_MAX_NCOLS columns, with no channels/samples and no ids. Gate/bias fusion IS
    // handled here at ncols_dst == 1: the FFN gate+up projection is the largest matmul in a
    // ternary decode step, and excluding it sent it to the generic mmvq kernel at ~2.3x the cost
    // (measured on gfx1100, Q2_B3: m=17408 68.9 us generic vs 30.1 us for the equivalent Vulkan
    // int-dot path). Note that comparison is batch-1: at ncols_dst == 8 the generic kernel
    // already hoists the weight load and its decode out of its column loop, so the batch-N
    // instantiation buys decode ALU, NOT weight bandwidth.
    // Gate/bias fusion and batch-N are mutually exclusive, not merely unimplemented together:
    // the generic broadcast bias read x_bias[j*stride_col_dst + ...] is in bounds only for
    // j == 0. That exclusion is enforced here, at the one dispatch gate.
    // x_scale/gate_scale remain excluded: they are asserted NVFP4-only upstream, so a
    // ternary tensor can never carry them, and the kernel implements no scale term.
    // Anything not covered falls through to the generic path unchanged.
    // q2_0 is an upstream type with its own tuned mmvq path, so the RDNA3 check is what keeps
    // this fork kernel from displacing it on hardware where it was never measured.
    if ((type_x == GGML_TYPE_Q2_0 || type_x == GGML_TYPE_Q2_B3)
            && ncols_dst >= 1 && ncols_dst <= TQ_TERNARY_MAX_NCOLS && ids == nullptr
            && nchannels_x == 1 && nchannels_y == 1 && nchannels_dst == 1
            && nsamples_x == 1 && nsamples_dst == 1
            && fusion.x_scale == nullptr && fusion.gate_scale == nullptr
            && (ncols_dst == 1
                || (!tq_ternary_batch_n_disabled()
                    && fusion.gate == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr))
            && !(tq_ternary_fusion_disabled()
                 && (fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr))
            && GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        // fusion.glu_op is only meaningful when a gate is present (the struct leaves it
        // default-initialized otherwise), so do not read it unless there is one.
        const ggml_glu_op glu_op = fusion.gate != nullptr ? fusion.glu_op : GGML_GLU_OP_SWIGLU;
        mul_mat_vec_q_ternary_decode(vx, type_x, fusion.gate, vy, dst,
            (const float *) fusion.x_bias, (const float *) fusion.gate_bias, glu_op,
            ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, stream);
        return;
    }
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_B3:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_B3>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_nelem  = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    // Reuse an activation vector already quantized for an earlier matmul in this same graph.
    // Restricted to the narrow batches over the ternary types: that is where the redundancy is
    // (432 matmuls / 257 distinct src1 nodes per decode token, and the speculative verify step
    // repeats it over TQ_Q8_1_CACHE_MAX_NE11 columns), and it bounds the retained pool memory to
    // ~257 * 5.5 KB per column. Prefill keeps the original per-call allocation, where the buffers
    // are large and each src1 is typically consumed once.
    const bool q8_1_cacheable = !tq_no_q8_1_cache()
        && (src0->type == GGML_TYPE_Q2_0 || src0->type == GGML_TYPE_Q2_B3)
        && ids == nullptr && ne11 <= TQ_Q8_1_CACHE_MAX_NE11 && ne12 == 1 && ne13 == 1;

    char * src1_q8_1_ptr = nullptr;
    ggml_cuda_pool_alloc<char> src1_q8_1_local;

    if (q8_1_cacheable) {
        for (const auto & e : ctx.q8_1_cache) {
            // Node identity, quantization target type and row length are what make the CONTENTS
            // reusable; the width and the allocated size are what make the BUFFER usable. An
            // entry too narrow for this consumer falls through to a fresh quantization -- it is
            // never evicted, because a narrower consumer may still be pending on it.
            if (e.src1 == src1 && e.src0_type == src0->type && e.ne10 == ne10
                    && e.ne11 == ne11 && e.alloc_nelem >= q8_1_nelem) {
                src1_q8_1_ptr = e.buf;
                break;
            }
        }
    }

    if (src1_q8_1_ptr == nullptr) {
        char * buf;
        if (q8_1_cacheable) {
            // Held in the context so the allocation outlives this call and stays valid for
            // the rest of the graph; released by q8_1_cache_reset() at the next graph.
            ctx.q8_1_cache_allocs.emplace_back(new ggml_cuda_pool_alloc<char>(ctx.pool(), q8_1_nelem));
            buf = ctx.q8_1_cache_allocs.back()->get();
            ctx.q8_1_cache.push_back({ src1, src0->type, ne10, ne11, q8_1_nelem, buf });
        } else {
            src1_q8_1_local.alloc(ctx.pool(), q8_1_nelem);
            buf = src1_q8_1_local.get();
        }

        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, buf, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        src1_q8_1_ptr = buf;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1_ptr, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
