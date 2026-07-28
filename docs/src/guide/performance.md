```@meta
CurrentModule = HuggingFaceTransformers
```

# Precision, quantization, and GPU

## Lower precision

[`Models.convert_eltype`](@ref) walks a model and converts every floating-point
array, returning a new model and leaving the original untouched.
[`Models.fp16`](@ref), [`Models.bf16`](@ref), and [`Models.fp32`](@ref) are the
shorthand.

```julia
lm16 = fp16(lm)
lmbf = bf16(lm)
back = fp32(lm16)
```

The docstring on `convert_eltype` records the tolerance band to expect per
dtype. Deeper models accumulate more error, so treat per-model low-precision
parity as something to measure rather than assume: re-record a fixture with
`torch_dtype=torch.float16` and compare it against an `fp16(model)` run.

Note that `Float32(-1e9)` mask constants and Gemma's softcap divisor stay in
Float32 regardless, which is deliberate.

## Weight-only int8

[`Models.quantize_int8`](@ref) replaces every [`Layers.Linear`](@ref) weight
with a [`Layers.QuantizedInt8Matrix`](@ref), a per-row symmetric int8 matrix.

```julia
lmq = quantize_int8(lm)
```

Norms and token embeddings stay Float32. Resident weights shrink about 4x, and
the matmul dequantizes per call, which trades compute for memory. Expect logit
error roughly in the `1e-2` to `1e-1` range against the fp32 reference, growing
with depth.

## GPU

Move a model with Flux and the rest follows:

```julia
using Flux: gpu
using CUDA   # or AMDGPU, or Metal on Apple Silicon

lm_gpu = gpu(lm)
```

Importing a backend loads the matching package extension, which routes
[`Layers.sdpa`](@ref) on device arrays to [`Layers.flash_sdpa`](@ref). That is a
tiled attention kernel: it walks the key and value blocks with a running softmax
instead of materializing the full score matrix, so memory stays linear in
sequence length. It also stays NaN-safe for a fully-masked query and preserves
input precision, so fp16 in gives fp16 out.

Three extensions ship, one per backend:

| Backend | Extension | Array type |
| --- | --- | --- |
| CUDA | `HuggingFaceTransformersCUDAExt` | `CuArray` |
| AMDGPU | `HuggingFaceTransformersAMDGPUExt` | `ROCArray` |
| Metal | `HuggingFaceTransformersMetalExt` | `MtlArray` |

Metal has no on-device Float64, so run in fp32 or fp16 there.

The CPU suite checks the tiled path against the materialized one on causal,
sliding-window, softcap, and multi-block inputs. To run the same comparison on a
real device:

```
HFT_GPU_BACKEND=cuda julia --project=. test/gpu_attention.jl
```

That file is not part of the default suite. See the [GPU CI plan](@ref
"GPU CI plan") for how automated GPU coverage is meant to land.

## Memory notes

A few allocations are sized for correctness rather than minimum footprint:

- Sliding-window models allocate `max_seq` cache slots regardless of the window,
  so a ring buffer capped at `min(max_seq, sliding_window)` would save memory.
- Tied embeddings materialize the transpose instead of sharing storage, which
  keeps the loader pure-data at a cost of roughly `vocab * hidden`.
- Mixtral batches tokens per expert but does not fuse the expert matmuls.
