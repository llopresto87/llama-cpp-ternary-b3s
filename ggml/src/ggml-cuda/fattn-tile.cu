#include "common.cuh"
#include "fattn-tile.cuh"

// TEMPORARY (§16.1 Step 3 proof-of-execution). Counts which of the two DKQ=DV=256 tile
// branches actually ran, and the deepest cache each saw, so that a benchmark cannot be
// believed until the changed path is shown to have executed on the measured shape
// (crosscut.rdna3-perf, perf.pass-discipline). Entirely inert unless TQ_HIP_FA_TRACE is
// set: the getenv is read once into a static and the counters are plain ints on the host
// dispatch path, which already does far more work per call than this.
struct tq_fa_tile_trace {
    long long n_native = 0, n_mirror = 0;
    long long deepest_native = 0, deepest_mirror = 0;
    ~tq_fa_tile_trace() {
        if (getenv("TQ_HIP_FA_TRACE") == nullptr) {
            return;
        }
        fprintf(stderr, "TQ_FA_TRACE: tile256 native_kv calls=%lld maxK=%lld | f16_mirror calls=%lld maxK=%lld\n",
                n_native, deepest_native, n_mirror, deepest_mirror);
    }
};
static tq_fa_tile_trace g_tq_fa_tile_trace;

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            // Read a q4_0 cache natively where that instance exists, so that launch_fattn
            // does not rebuild an f16 mirror of the whole cache on every call. The
            // predicate is shared with ggml_cuda_flash_attn_ext_get_alloc_size; they must
            // not be allowed to drift apart.
            if (ggml_cuda_fattn_tile_use_native_kv(cc, K, V)) {
                g_tq_fa_tile_trace.n_native++;
                g_tq_fa_tile_trace.deepest_native = std::max(g_tq_fa_tile_trace.deepest_native, (long long) K->ne[1]);
                ggml_cuda_flash_attn_ext_tile_case<256, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>(ctx, dst);
            } else {
                g_tq_fa_tile_trace.n_mirror++;
                g_tq_fa_tile_trace.deepest_mirror = std::max(g_tq_fa_tile_trace.deepest_mirror, (long long) K->ne[1]);
                ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
            }
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}

