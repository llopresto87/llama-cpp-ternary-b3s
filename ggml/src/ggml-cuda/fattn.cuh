#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);


// quantity the production launch site derives passed in explicitly.
//
// bound-record stride the production path can only reach by having a real KV
// cache in a particular state: `n_pages_stride` is a SEPARATE argument from
// `n_pages` precisely so the test can pass the wrong one and prove its own
// check has teeth. Production always passes bounds->ne[1].
//
        cudaStream_t stream,
        int * KV_pages, const void * bounds, const char * Q, const void * mask_row,
        int n_lists, int list_stride, int list_off,
        int n_pages, int n_pages_stride,
        int n_kv, int n_channels, int n_head_kv,
        int page_size, int block_tokens, int n_blocks,
        int top_k, int n_sink_pages, int window,
        int n_head, size_t nb01, size_t nb02, float scale);
