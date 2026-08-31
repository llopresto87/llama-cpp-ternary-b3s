# Q2_B3 ("B3S"): Ternary Bonsai Base-3 Quantization

This fork of llama.cpp adds `GGML_TYPE_Q2_B3` — a ternary (3-level) block
quantization format designed for weights that are already ternary-native
(BitNet-b1.58-style / "Ternary Bonsai" models). The format is an alternative to
the upstream ternary types `TQ1_0`/`TQ2_0`: it packs 128 weights per block in
base-3 at 1.75 bits per weight, and — unlike the upstream TQ types — ships GPU
kernels for CUDA/HIP, Vulkan, and Metal in this tree.

All identifiers, constants, and byte counts below were read from this checkout.

## 1. What is B3S / Q2_B3

`Q2_B3` is a ggml quantization type (`GGML_TYPE_Q2_B3 = 43`) and a GGUF/llama
file type (`LLAMA_FTYPE_MOSTLY_Q2_B3 = 42`, `GGML_FTYPE_MOSTLY_Q2_B3 = 29`)
that stores each weight as one of exactly three values — `-1`, `0`, or `+1` —
scaled by a single per-block f16 scale. Instead of writing 2 bits per weight as
binary two-bit codes, the three states are packed in base 3, five trits per
byte (`3^5 = 243 <= 256`), which gets the payload below 2 bits per weight and
close to the information-theoretic floor of `log2(3) ≈ 1.585` bits per ternary
symbol. On a 128-weight block the total cost is 28 bytes (26 bytes of trits +
2-byte f16 scale) = **1.75 bits per weight**.

Source: `ggml/include/ggml.h:433,479`, `include/llama.h:159`,
`ggml/src/ggml-common.h:197`.

## 2. Format details

### Block structure

`QK2_B3 = 128` weights per block. One block is:

```c
#define QK2_B3 128
typedef struct {
    ggml_half d;            // single scale for all 128 values
    uint8_t qs[26];         // base-3 packed, v2 chunk-aligned (see quantize_row_q2_b3_ref)
} block_q2_b3;
static_assert(sizeof(block_q2_b3) == sizeof(ggml_half) + 26, "wrong q2_b3 block size/padding");
```

(`ggml/src/ggml-common.h:197-201`). The ggml type table registers it as
quantized, block size 128, `sizeof` 28 bytes, with `dequantize_row_q2_b3` /
`quantize_row_q2_b3_ref` as the to/from-float paths
(`ggml/src/ggml.c:691-698`).

### Bits per weight

```
28 bytes/block × 8 bits/byte = 224 bits/block
224 bits / 128 weights       = 1.75 bits per weight
```

Breakdown: 26 bytes (208 bits) carry the 128 trits → 1.625 bpw payload; the
2-byte f16 scale adds 0.125 bpw → 1.75 bpw total. The payload itself is about
2.5% above the `log2(3) ≈ 1.585` floor (128 trits × 1.585 ≈ 203 bits fit in
208).

### Ternary value set

Dequantization (`ggml/src/ggml-quants.c:474-494`) is:

```c
const int q = (x[i].qs[byte_i] / pw3[digit]) % 3;
y[i*qk + j] = (q - 1) * d;   /* q ∈ {0,1,2} → weight ∈ {-1,0,+1} × d */
```

so each weight is exactly one of `{-d, 0, +d}` — symmetric, no zero-point.
Quantization (`quantize_row_q2_b3_ref`, `ggml-quants.c:439-472`) computes the
block absmax `amax = max|x|`, stores `d = fp16(amax)`, and rounds each weight
to the nearest trit (`q = round(w/d) + 1`, clamped to [0,2]). Single f16 scale
per 128 weights, absmax-scaled, no importance-matrix weighting in this codec
path (`quantize_q2_b3` at `ggml-quants.c:2185-2198` performs the same ref
rounding whether or not an imatrix is passed).

### Base-3 packing ("v2 chunk-aligned")

The 128 trits of a block are split into 4 chunks of 32 trits (`QI2_B3 =
QK2_B3/32 = 4`; `ggml-common.h:102`). Each chunk `c` owns:

- bytes `[6c .. 6c+5]` — 30 trits at 5 trits/byte (digits 0..4), and
- 2 "straggler" trits in the shared byte `24 + (c>>1)` at digit offset
  `2*(c&1)` (trits 30 and 31 of the chunk).

```c
const int c = j >> 5, t = j & 31;
const int byte  = t < 30 ? 6*c + t/5      : 24 + (c >> 1);
const int digit = t < 30 ? t % 5          : 2*(c & 1) + (t - 30);
```

(`ggml-quants.c:466-469`, mirrored by the CPU vec-dot at
`ggml/src/ggml-cpu/quants.c:264-267`). So bytes 24 and 25 each hold 4
straggler trits (two from each of a chunk pair); every straggler value is
`< 3^2 = 9`, so there is no cross-byte carry. Because the byte and digit
offsets of every trit are compile-time constants per index, GPU kernels can
decode with shifts alone — per the in-tree comment, "constant per-chunk byte
offsets let the GPU mmvq decode with compile-time shifts (no runtime-skip u64
chain)" (`ggml-quants.c:455-458`).

The dot product against q8_0 activations implements the same decode: one
q2_b3 block maps to four q8_0 blocks, and each 32-trit chunk contributes
`sum += (trit-1) * qy[t]` before the scales are applied
(`ggml_vec_dot_q2_b3_q8_0_generic`, `ggml-cpu/quants.c:229-278`).

## 3. Advantages

- **Memory footprint.** At 1.75 bpw, weight storage is smaller than any 2-bit
  codec in this tree: Q2_0 here is 18 bytes per 64 weights = 2.25 bpw
  (`ggml-common.h:190-195`), and standard 4-bit types run ~4.5-5.5 bpw. For a
  27B model, 1.75 bpw is `27e9 × 1.75 / 8 ≈ 5.9 GB` of weights (derived from
  the bpw above, not benchmarked).
- **Multiplication-light dot product.** The weight side of the dot product is
  ternary, so multiplying a weight by an int8 activation reduces to
  `±activation` (add/subtract) or `0` (skip); the block scale `d` is applied
  once per block after integer accumulation, with the q8_0 chunk scale applied
  per 32-chunk as usual (`ggml-cpu/quants.c:252-275`). On CUDA the MMQ path
  decodes the trits into a q8_0-shaped int8 tile and reuses q8_0's vec_dot
  (`ggml/src/ggml-cuda/vecdotq.cuh:115-116`).
- **Real GPU kernels, unlike upstream TQ types.** This fork ships:
  - CUDA/HIP: an MMQ instance (`ggml-cuda/mmq-instance-q2_b3.cu`, configs in
    `mmq-config-rdna3.cuh` / `mmq-config-rdna4.cuh`), MMVQ vector kernels
    (`ggml-cuda/mmvq.cu`) including a bespoke RDNA3 (gfx1100) ternary decode
    GEMV with per-type wave tuning (`mmvq.cu:1096-1403`).
  - Vulkan: `dequant_q2_b3.comp` plus Q2_B3 mul-mat / mul-mat-vec shader
    wiring (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`).
  - Metal (Apple): `dequantize_q2_b3` plus `mul_mm` / `mul_mv` / `cpy` /
    `get_rows` kernels and device dispatch
    (`ggml/src/ggml-metal/kernels/{dequantize.h,mul_mm.metal,mul_mv.metal,quantize.metal}`,
    `ggml-metal-device.cpp`/`.m`).

  By contrast, upstream ternary `TQ1_0`/`TQ2_0` have **no CUDA/HIP code**: a
  grep of `ggml/src/ggml-cuda` in this tree for `tq1_0|tq2_0` returns zero
  matches, so those types are CPU-only in CUDA builds.
- **Near-lossless on ternary-native weights.** For models whose weights are
  already ternary up to a per-block scale (BitNet-b1.58-style, Ternary Bonsai),
  Q2_B3 quantization reproduces each weight exactly (the `round(w/d)` step is
  an identity) up to the f16 rounding of the block scale itself. The codec is
  not a general low-bit compressor (see below); its fidelity argument applies
  specifically to this model family.

## 4. Limitations / honest caveats

- **GPU kernels are UNVERIFIED on hardware.** The CUDA/HIP, Metal, and Vulkan
  Q2_B3 kernels compile, but have **not** been confirmed to produce correct
  output on real GPUs in this tree. Only the CPU reference codec is known-good.
  Treat all GPU paths as experimental: for trustworthy output right now, run on
  CPU. Correctness reports for any GPU backend are exactly the feedback wanted.
- **Only for ternary-native weights.** Q2_B3 is a ternary codec, not a general
  2-bit quantizer for arbitrary FP models. Applying it to ordinary FP16
  weights collapses each weight to three levels; quality on non-ternary models
  is expected to degrade sharply. Verdict on real models: not measured here —
  this tree contains no perplexity benchmark results for random-FP models in
  Q2_B3.
- **Single scale per 128-block.** One f16 absmax scale for all 128 weights,
  no per-sub-block scales and no zero-point. Fine for bimodal/symmetric
  ternary weights; wasteful or inaccurate for skewed distributions.
- **Payload above the floor.** 1.625 bpw of trit payload sits ~2.5% above
  `log2(3)`, and the shared scale adds 0.125 bpw — hence 1.75 bpw total vs the
  ~1.585-bit ideal. Upstream TQ1_0 (labeled 1.69 bpw in
  `tools/quantize/quantize.cpp:48`) is denser on paper but has no CUDA/HIP
  kernels.
- **Throughput numbers:** not measured here. The tree contains tuning notes
  for Ternary-Bonsai-27B Q2_B3 on RX 7900 XTX (gfx1100), e.g. MMVQ wave and
  Vulkan row-multiplier sweeps (`mmvq.cu:1396-1403`,
  `ggml-vulkan.cpp:5350-5357`), but no reproducible benchmark suite or
  perplexity measurements to cite.

## 5. Usage pointer

Q2_B3 is a drop-in ggml type: it plugs into standard llama.cpp model loading
and inference through the regular ggml type table (`ggml.c:691-698`) and the
llama ftype mapping (`src/llama-model-loader.cpp:42,776`). Models carrying
Q2_B3 tensors load and run with the normal llama-cli / llama-server paths; no
extra runtime switch is required.

On the Python side the GGUF registration lives in `gguf-py/gguf/constants.py`:

- `GGMLQuantizationType.Q2_B3 = 43` (line 5504),
- `GGMLQuantizationType.MOSTLY_Q2_B3 = 42` (line 5561),
- block geometry `Q2_B3: (128, 2 + 26)` (line 5698).

GGUF files with the Q2_B3 ftype/tensors are therefore recognized by the
standard `gguf-py` tooling in this repo.

### Producing a Q2_B3 model (repacker)

The tool that converts an already-ternary checkpoint into a `Q2_B3` GGUF is
maintained in its own repository:

- **Repacker:** https://github.com/llopresto87/ternary-q2_0-repacker

Point it at a ternary-native checkpoint (e.g. a Ternary-Bonsai model) to emit a
`Q2_B3` GGUF, then load that GGUF with the `llama-cli` / `llama-server` built
from this fork. (Reminder: for now, run inference on the CPU backend — the GPU
kernels are not yet hardware-verified.)
