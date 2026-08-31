#include "llama-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

void llama_hparams::set_swa_pattern(uint32_t n_pattern, bool dense_first) {
    if (dense_first) {
        for (uint32_t il = 0; il < n_layer(); ++il) {
            is_swa_impl[il] = n_pattern == 0 || (il % n_pattern != 0);
        }
    } else {
        for (uint32_t il = 0; il < n_layer(); ++il) {
            is_swa_impl[il] = n_pattern == 0 || (il % n_pattern < (n_pattern - 1));
        }
    }

    for (uint32_t il = n_layer(); il < n_layer_all; ++il) {
        is_swa_impl[il] = false;
    }
}

void llama_hparams::set_recr_pattern(uint32_t n_pattern, bool dense_first) {
    if (dense_first) {
        for (uint32_t il = 0; il < n_layer(); ++il) {
            is_recr_impl[il] = n_pattern == 0 || (il % n_pattern != 0);
        }
    } else {
        for (uint32_t il = 0; il < n_layer(); ++il) {
            is_recr_impl[il] = n_pattern == 0 || (il % n_pattern < (n_pattern - 1));
        }
    }

    for (uint32_t il = n_layer(); il < n_layer_all; ++il) {
        is_recr_impl[il] = false;
    }
}

bool llama_hparams::is_swa_any() const {
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        if (is_swa_impl[il]) {
            return true;
        }
    }

    return false;
}

    for (uint32_t il = 0; il < n_layer_all; ++il) {
            return true;
        }
    }

    return false;
}

// the minimum window that is still a window; anything shorter is far more likely a typo
// than an intent, and no published result windows a 262144-context model this tightly
static constexpr uint32_t LLAMA_SWA_MIN_WINDOW = 256;

// Parse a non-negative decimal integer. Deliberately stricter than strtoul: a leading '-',
// surrounding space, a trailing unit and an empty value are all rejected rather than
// silently becoming a number the operator did not write. In particular `TQ_SWA_WINDOW=`
// (set but empty, which getenv reports as "" and not as unset) is malformed, not unset - it
// is what an unexpanded shell variable looks like, and reading it as "off" would run full
// attention under a window the operator believes is in force.
static bool llama_swa_parse_u32(const std::string & val, uint32_t & out) {
    if (val.empty()) {
        return false;
    }

    uint64_t v = 0;
    for (const char c : val) {
        if (c < '0' || c > '9') {
            return false;
        }

        v = v*10 + (uint64_t) (c - '0');
        if (v > UINT32_MAX) {
            return false;
        }
    }

    out = (uint32_t) v;

    return true;
}

// the same parse for a knob that is a bare number, with the variable and the value named in
// the error - the loader has no other way to tell the operator which one it rejected
static uint32_t llama_swa_knob_u32(const char * var, const std::string & val) {
    uint32_t out = 0;

    if (!llama_swa_parse_u32(val, out)) {
        throw std::runtime_error(std::string(var) + "=\"" + val + "\" is not a non-negative integer");
    }

    return out;
}

void llama_hparams::apply_swa_knobs(const char * env_window, const char * env_sink, const char * env_layers) {
    // Nothing is written before every knob has been parsed, resolved and validated, so a
    // rejected configuration leaves the model exactly as it was.
    const uint32_t n_swa_new = env_window ? llama_swa_knob_u32("TQ_SWA_WINDOW", env_window) : 0;

    if (n_swa_new == 0) {
        // the whole feature is off, so the other two knobs cannot mean anything
        if (env_sink) {
            throw std::runtime_error(std::string("TQ_SWA_SINK=\"") + env_sink +
                    "\" requires TQ_SWA_WINDOW > 0");
        }

        if (env_layers) {
            throw std::runtime_error(std::string("TQ_SWA_LAYERS=\"") + env_layers +
                    "\" requires TQ_SWA_WINDOW > 0");
        }

        return;
    }

    if (n_swa_new < LLAMA_SWA_MIN_WINDOW) {
        throw std::runtime_error("TQ_SWA_WINDOW=" + std::to_string(n_swa_new) +
                " is below the minimum window of " + std::to_string(LLAMA_SWA_MIN_WINDOW) + " tokens");
    }

    if (n_swa_new >= n_ctx_train) {
        throw std::runtime_error("TQ_SWA_WINDOW=" + std::to_string(n_swa_new) +
                " must be smaller than the trained context of " + std::to_string(n_ctx_train) +
                " tokens; a window that spans the context masks nothing");
    }

    const uint32_t n_sink_new = env_sink ? llama_swa_knob_u32("TQ_SWA_SINK", env_sink) : 0;

    if (n_sink_new >= n_swa_new) {
        throw std::runtime_error("TQ_SWA_SINK=" + std::to_string(n_sink_new) +
                " must be smaller than TQ_SWA_WINDOW=" + std::to_string(n_swa_new) +
                "; a sink region as large as the window is not a window");
    }

    // The attention layers in ascending order. Read from is_recr_impl[], which the loader
    // fills from the GGUF's recurrent-layer list when it has one - re-deriving them from a
    // layer interval would disagree with the model whenever that list is present.
    std::vector<uint32_t> attn;
    for (uint32_t il = 0; il < n_layer(); ++il) {
        if (!is_recr(il)) {
            attn.push_back(il);
        }
    }

    // `even`/`odd` are over the attention-layer ordinal k, never over il: on a hybrid whose
    // attention layers are {3, 7, ..., 63} every il is odd, so an absolute reading would
    // select all of them or none of them.
    const std::string layers = env_layers ? env_layers : "all";

    std::vector<uint32_t> selected;
    if (layers == "all" || layers == "even" || layers == "odd") {
        const uint32_t k_parity = (layers == "odd") ? 1 : 0;
        for (uint32_t k = 0; k < attn.size(); ++k) {
            if (layers == "all" || k % 2 == k_parity) {
                selected.push_back(attn[k]);
            }
        }
    } else if (!layers.empty()) {
        // an explicit list of absolute layer indices
        size_t pos = 0;
        while (pos <= layers.size()) {
            const size_t      end = std::min(layers.find(',', pos), layers.size());
            const std::string tok = layers.substr(pos, end - pos);

            uint32_t il = 0;
            if (!llama_swa_parse_u32(tok, il)) {
                throw std::runtime_error("TQ_SWA_LAYERS=\"" + layers + "\": \"" + tok +
                        "\" is neither a layer index nor one of \"all\", \"even\", \"odd\"");
            }

            if (il >= n_layer()) {
                throw std::runtime_error("TQ_SWA_LAYERS=\"" + layers + "\": layer " + std::to_string(il) +
                        " is out of range; the model has " + std::to_string(n_layer()) + " layers");
            }

            if (is_recr(il)) {
                // filter_attn drops recurrent layers from both KV caches, so windowing one
                // would do nothing except flip is_swa_any() and re-base its shift-rope
                throw std::runtime_error("TQ_SWA_LAYERS=\"" + layers + "\": layer " + std::to_string(il) +
                        " is a recurrent (gated DeltaNet) layer and has no KV cache to window");
            }

            selected.push_back(il);
            pos = end + 1;
        }
    }

    if (selected.empty()) {
        throw std::runtime_error("TQ_SWA_LAYERS=\"" + layers +
                "\" selects no layers; a window with no windowed layer is inert");
    }

    // commit
    n_swa    = n_swa_new;
    n_sink   = n_sink_new;
    swa_type = LLAMA_SWA_TYPE_STANDARD;

    std::fill(is_swa_impl.begin(), is_swa_impl.end(), 0);
    for (const uint32_t il : selected) {
        is_swa_impl[il] = 1;
    }

    // llama_model::get_rope_freq_base/_scale route every windowed layer to these two
    // fields, whose defaults (10000.0f / 1.0f) are unrelated to any given model's training.
    // Without this the first context shift rotates the windowed layers with a wrong base.
    rope_freq_base_train_swa  = rope_freq_base_train;
    rope_freq_scale_train_swa = rope_freq_scale_train;
}

void llama_hparams::validate_swa() const {
    // attention sinks are only honored by the standard window (see is_masked_swa), so a
    // sink under any other geometry would be silently dropped
    if (n_sink > 0 && swa_type != LLAMA_SWA_TYPE_STANDARD) {
        throw std::runtime_error("n_sink = " + std::to_string(n_sink) +
                " requires the standard sliding window (swa_type = " + std::to_string((int) swa_type) + ")");
    }

    if (n_sink > 0 && n_sink >= n_swa) {
        throw std::runtime_error("n_sink = " + std::to_string(n_sink) +
                " must be smaller than n_swa = " + std::to_string(n_swa));
    }

    if (swa_type == LLAMA_SWA_TYPE_NONE && is_swa_any()) {
        throw std::runtime_error("a layer is marked SWA but swa_type is none");
    }

    for (uint32_t il = n_layer(); il < n_layer_all; ++il) {
        if (is_swa_impl[il]) {
            throw std::runtime_error("layer " + std::to_string(il) +
                    " is marked SWA but is beyond the " + std::to_string(n_layer()) + " effective layers");
        }
    }
}

// of the sparse-capable sets measured at needle depths 25, 50 and 75, in attention-layer
// ordinals, so it survives any change to the attention interval. It is a measured default
// and not a property of the model (ADR-0017): a layer wrongly marked sparse silently drops
// content at some depth, while one wrongly left dense costs only the speedup it would have
// contributed, so the default is the intersection and never the union.
//
// Amendment 2). Ordinals 13 and 14 (il = 55, 59) were in the v0.2.1 default and are out on
// the intersection rule above: they retain only ~50% of the attention mass even at PERFECT
// selection rank, so no value of K buys them back. Ordinal 7 (il = 31) was dropped one

// B = 64, settled on measured data 2026-07-30 (was 32).
//
// It is the only ladder value with ZERO over-fetch against the kernel's 64-token
// actually needs -- its coverage TIES B=32 at 89.9% while costing 11.6% fewer
// total bytes. B=32's ranking edge is real at small K and vanishes exactly where
// the hard case lives. Do not move this back without re-running OQ-D at EQUAL
// REALIZED LOADED BYTES; at equal K*B the smaller page is charged nothing for
// its over-fetch, which is the comparison that picks the wrong B.

// B must tile the existing device-side KV skip structure, whose block is
// FATTN_KQ_STRIDE = 256 tokens; a page that straddles two blocks has no kernel.


// split "a,b,c" on commas WITHOUT collapsing empty fields: "8,,16" has an empty rung and is
// malformed, and a splitter that drops it would silently accept the operator's typo
    std::vector<std::string> out;

    size_t pos = 0;
    while (pos <= val.size()) {
        const size_t end = std::min(val.find(',', pos), val.size());
        out.push_back(val.substr(pos, end - pos));
        pos = end + 1;
    }

    return out;
}

// the attention layers in order, so that index k IS the attention-layer ordinal the
// from full_attention_interval: a GGUF that lists its recurrent layers explicitly can
// disagree with the (il+1)%n fallback. One home, because the knob resolves ordinal -> il
// and the §4.R banner has to invert it.
    std::vector<uint32_t> attn;

    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (!hp.is_recr(il)) {
            attn.push_back(il);
        }
    }

    return attn;
}

    std::string out;

    for (const uint32_t v : vals) {
        if (!out.empty()) {
            out += ",";
        }
        out += std::to_string(v);
    }

    return out;
}

        const char * env_k,
        const char * env_ladder,
        const char * env_page,
        const char * env_layers,
        const char * env_window,
        const char * env_sink,
        const char * env_prefill,
        const char * env_stats) {
    // As in apply_swa_knobs: nothing is written before every knob has been parsed, resolved
    // and validated, so a rejected configuration leaves the model exactly as it was.
    if (!env_k) {
        // separate enable flag would admit "enabled with no K", which has no meaning.
        const struct { const char * name; const char * val; } others[] = {
        };

        for (const auto & o : others) {
            if (o.val) {
            }
        }

        return;
    }

    // A window and a content selection are two different answers to the same question about
    // the same layers, and a run under both would be uninterpretable.
    if (swa_type != LLAMA_SWA_TYPE_NONE || n_swa != 0) {
                "they are two different answers to the same question about the same layers");
    }

    // --- the ladder: the set of K values the kernel is COMPILED for ----------------------
    // It does not quantize an adaptive count - it enumerates the shapes that exist, so a K
    // off the ladder has no kernel and is a load failure rather than a rounded value.

    std::vector<uint32_t> ladder;
        uint32_t rung = 0;
        if (!llama_swa_parse_u32(tok, rung)) {
                    "\" is not a rung; a rung is a bare non-negative decimal integer");
        }

        if (rung == 0) {
                    "\": a rung of 0 selects no distant page and is inexpressible as an intent");
        }

        if (!ladder.empty() && rung <= ladder.back()) {
        }

        ladder.push_back(rung);
    }

                std::to_string(ladder.size()) + " rungs; at most " +
                "kernel specialization");
    }

    // --- K, the master switch -------------------------------------------------------------

    if (k_new == 0) {
    }

    if (std::find(ladder.begin(), ladder.end(), k_new) == ladder.end()) {
                "; no kernel specialization exists for it");
    }

    // --- B, a kernel tiling parameter and not a free dial ---------------------------------

    }

    // --- the always-resident set ----------------------------------------------------------
    // W = 0 and sink = 0 are legal and mean "content selection only"; they are not the
    // default because the measured output error is markedly lower with a window.

    if (window_new >= n_ctx_train) {
                " must be smaller than the trained context of " + std::to_string(n_ctx_train) +
                " tokens; a window that spans the context leaves no distant page to select");
    }


    }

    if (window_new > 0 && sink_new >= window_new) {
                "; a sink region as large as the window is not a window");
    }

    // --- the two boolean knobs -------------------------------------------------------------

    if (prefill_new > 1) {
    }


    if (stats_new > 1) {
    }

    // --- the layer set, over the ATTENTION-LAYER ORDINAL ------------------------------------
    // Same resolution apply_swa_knobs uses, and for the same reason - but here it governs the
    // explicit list too, not only even/odd: the operator configures in ordinals, and 63 is a
    // valid il and an invalid ordinal, so reading a list over il would accept the single most
    // likely operator error and silently select nothing.


    std::vector<uint32_t> selected;
    if (layers == "all" || layers == "even" || layers == "odd") {
        const uint32_t k_parity = (layers == "odd") ? 1 : 0;
        for (uint32_t k = 0; k < attn.size(); ++k) {
            if (layers == "all" || k % 2 == k_parity) {
                selected.push_back(attn[k]);
            }
        }
    } else {
            uint32_t ord = 0;
            if (!llama_swa_parse_u32(tok, ord)) {
                        "\" is neither an attention-layer ordinal in [0," + std::to_string(attn.size()) +
                        ") nor one of \"all\", \"even\", \"odd\"");
            }

            if (ord >= attn.size()) {
                        " is out of the attention-layer ordinal range [0," + std::to_string(attn.size()) +
                        "); it is read as an ORDINAL, not as an absolute layer index");
            }

            selected.push_back(attn[ord]);
        }
    }

    if (selected.empty()) {
                "\" selects no layer; an empty layer set is not \"off\"");
    }

    // commit


    for (const uint32_t il : selected) {
    }

    // must stay resident to be selectable, and an iswa-sized cache evicts exactly the pages
    // the mechanism exists to find (§4.M).
}

//
// "The banner is the run's own testimony; an operator's assertion that they exported a
// variable is not evidence." Before it existed, the only witness to which configuration ran
// was the KV-buffer size - 2344 MiB for core-5 against 2360 for core-7 at the operating
// point - which is a mapping the reader had to already know, and which says nothing at all
// about K, B, the window or the sink. That is how the layer-set default diverged from the
// spec across two versions with every result in between recorded as "at the default" (§7.V).
//
// The content is separated from the emission on purpose: this is a pure function of the
// resolved hparams, so it is assertable in the no-GPU Group 1 suite
// one environment read. A banner derived where nothing can test it satisfies the log and
// not the contract.
    std::vector<std::pair<std::string, std::string>> lines;

    // §4.R last clause: with the knobs unset, none of these lines is printed.
        return lines;
    }

    // BOTH readings of the layer set. The knob is written in attention-layer ordinals and
    // the run sparsifies absolute il values; the two lists are both short comma-separated
    // may be "even", "all", or absent entirely - looks correct while attesting nothing.

    std::vector<uint32_t> ordinals;
    std::vector<uint32_t> absolute;
    for (uint32_t k = 0; k < attn.size(); ++k) {
            ordinals.push_back(k);
            absolute.push_back(attn[k]);
        }
    }


    // Both end-to-end gates read these lines, so the shape is part of the contract: one
    // field per line, a lowercase key, and a value that is a bare integer or a comma list.

    // (§3.2.1). A gate that reads a retrieval score off a degenerate run reports a result
    // the mechanism did not earn.

    return lines;
}

//
// The contract's first clause is already true, and NOT because of this knob: nothing reads
// (ggml/src/ggml-cuda/fattn-common.cuh) requires cols_per_block == 1 && Q->ne[1] == 1, so a
// prefill launch falls back to the stock kernel, and because launch_fattn defaults
//
// What was missing is the SAYING of it. "The fallback is announced once per load, not
// catalogues under perf.gates-that-lie, and §7.V is what it has already cost this spec once.
//
// Prose, not (key, value) pairs, and deliberately: scripts/tests/gate_retrieval_depth.py
// banner's shape would be absorbed into the attested configuration instead of read by a
// same reason - the exact §4.N string is assertable in a suite that loads no model.
    std::vector<std::string> lines;

        return lines;
    }

    lines.emplace_back(
            "FULLY and is bit-identical to baseline. Enforced by the kernel shape guard "

        // Verbatim, exactly as §4.N clause 3 writes it. Any evaluation that records a prefill
        // number must carry this line, so it is greppable and never a format string.
        lines.emplace_back(

        // And the fact the operator most needs and the line above does not give them: the
        // knob is accepted, range-checked and attested, and it still selects nothing. §3.6
        // specifies the prefill geometry; no kernel implements it, and experiment E-2
        // (§11 OQ-C) has not reported. Saying "UNVALIDATED" without saying "and inert" would
        // read as "on, but unmeasured".
        lines.emplace_back(
                "§3.6's prefill geometry and experiment E-2 (§11 OQ-C) has not reported; this "
                "run is full attention at prefill");
    }

    return lines;
}

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer_all) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer_all) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer_all) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head/n_head_kv;
}

uint32_t llama_hparams::n_rot(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa(il) ? n_rot_swa : n_rot_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_inp() const {
    if (n_embd_inp_impl > 0) {
        return n_embd_inp_impl;
    }

    uint32_t n_embd_inp = n_embd;

    if (n_deepstack_layers > 0) {
        n_embd_inp += n_embd * n_deepstack_layers;
    }

    return n_embd_inp;
}

uint32_t llama_hparams::n_embd_inp_enc() const {
    return n_embd_inp_enc_impl > 0 ? n_embd_inp_enc_impl : n_embd_inp();
}

uint32_t llama_hparams::n_embd_out() const {
    return n_embd_out_impl > 0 ? n_embd_out_impl : n_embd;
}

uint32_t llama_hparams::n_embd_head_k(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa(il) ? n_embd_head_k_swa : n_embd_head_k_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_head_v(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa(il) ? n_embd_head_v_swa : n_embd_head_v_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k(il) * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_v(il) * n_head_kv;
}

bool llama_hparams::is_n_embd_k_gqa_variable() const {
    const uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        if (val != n_embd_k_gqa(il)) {
            return true;
        }
    }

    return false;
}

bool llama_hparams::is_n_embd_v_gqa_variable() const {
    const uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        if (val != n_embd_v_gqa(il)) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_embd_k_gqa_max() const {
    uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        val = std::max(val, n_embd_k_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_v_gqa_max() const {
    uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        val = std::max(val, n_embd_v_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_r() const {
    if (wkv_head_size != 0) {
        // for RWKV models
        return token_shift_count * n_embd;
    }

    if (n_shortconv_l_cache != 0) {
        // for LFM2 models
        return n_embd * (n_shortconv_l_cache - 1);
    }

    if (n_embd_head_kda != 0) {
        // for Kimi KDA layers
        // Conv state for Q, K, V: 3 * (d_conv - 1) * n_head * head_dim
        const uint32_t d_inner = n_head() * n_embd_head_kda;  // 32 * 128 = 4096
        return 3 * (ssm_d_conv > 0 ? ssm_d_conv - 1 : 3) * d_inner;
    }

    // TODO: maybe support other convolution strides than 1
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    // Corresponds to Mamba's conv_states size
    const uint32_t n_conv = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state);

    // PLE conv history needs its own row: Meta splits cache_r_l by head, so a history packed behind the first is unaddressable
    // it lives in cache_ple_r_l instead, mirrored like the rest of the PLE module
    return n_conv;
}

uint32_t llama_hparams::n_embd_s() const {
    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    if (n_embd_head_kda != 0) {
        // for Kimi KDA layers
        // Full recurrent state: head_dim * head_dim * n_head
        // h tensor shape for delta attention: [head_dim, head_dim, n_head]
        return n_embd_head_kda * n_embd_head_kda * n_head();  // 128 * 128 * 32 = 524288
    }

    if (n_embd_head_la != 0) {
        // for MiniMax-Text-01 linear attention layers
        // Full recurrent state: head_dim * head_dim * n_head
        // tensor shape for linear attention: [head_dim, head_dim, n_head]
        return n_embd_head_la * n_embd_head_la * n_head();  // 128 * 128 * 64 = 1048576
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}

bool llama_hparams::is_recr(uint32_t il) const {
    if (il < n_layer_all) {
        return is_recr_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

uint32_t llama_hparams::ple_conv_state() const {
    if (ple_n_heads == 0 || ple_conv_kernel == 0) {
        return 0;
    }

    // dilation equals the n-gram size, matching the reference module
    return (ple_conv_kernel - 1) * ple_ngram_size * dsv4_hc_mult * n_embd;
}

bool llama_hparams::is_ple(uint32_t il) const {
    if (il < n_layer_all) {
        return is_ple_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

uint32_t llama_hparams::n_pos_per_embd() const {
    return rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
}

    if (il < n_layer_all) {
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

bool llama_hparams::is_swa(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

bool llama_hparams::is_mla() const {
    assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
           (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}

bool llama_hparams::is_indexer_full(uint32_t il) const {
    if (il < n_layer()) {
        return is_indexer_full_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer: %u)\n", __func__, il, n_layer());
}

uint32_t llama_hparams::n_embd_head_k_mla() const {
    return is_mla() ? n_embd_head_k_mla_impl : n_embd_head_k();
}

uint32_t llama_hparams::n_embd_head_v_mla() const {
    return is_mla() ? n_embd_head_v_mla_impl : n_embd_head_v();
}

bool llama_hparams::has_kv(uint32_t il) const {
    if (n_layer_kv_from_start >= 0) {
        if (il < (uint32_t) n_layer_kv_from_start) {
            return true;
        }

        return false;
    }

    // by default, all layers have kv
    return true;
}

bool llama_hparams::has_rope(uint32_t il) const {
    // the router layer stores adapter routing signal, not positional info,
    // so it must not be RoPE-shifted
    if (router_layer >= 0 && (int32_t) il == router_layer) {
        return false;
    }

    if (il < n_layer_all) {
        return rope_pattern[il] != 0;
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

uint32_t llama_hparams::n_layer() const {
    return n_layer_all - n_layer_nextn;
}

bool llama_hparams::use_mrope() const {
    return rope_sections[0] > 0 && rope_sections[1] > 0;
}
