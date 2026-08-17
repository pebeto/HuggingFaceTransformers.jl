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

Two questions are separate here: where the arrays live, and which attention
kernel runs.

Every forward pass is generic array math (`*`, `batched_mul`, `softmax`,
broadcast), so it runs on whichever GPUArrays backend you load, dispatching on
the array type. No vendor-specific code sits in the core, and a model does not
need a backend extension to execute on a device.

```julia
using Flux: gpu
using CUDA   # or AMDGPU, or Metal on Apple Silicon

lm_gpu = gpu(lm)
caches = build_caches(lm_gpu, 512, 1)
```

[`Models.build_caches`](@ref) allocates each cache on the model's device, so
cache writes land next to the activations instead of forcing a host round trip.

Every architecture in the package runs on device: the decoder LLMs including
Mixtral's expert routing and NeoX's partial rotary, the vision encoders, Whisper,
the embedding models, and the LLaVA composition. Training works on device too,
including LoRA through a checkpointed forward. The suite checks all of it against
JLArrays, the GPUArrays reference backend, so the device path stays covered
without a GPU in CI, and the same assertions run against CUDA, AMDGPU, or Metal
by setting `HFT_GPU_BACKEND`.

!!! warning "`gpu` can silently do nothing"
    `Flux.gpu` moves parameters only once an MLDataDevices trigger package is
    fully loaded. `using CUDA` without `cuDNN` leaves every parameter a plain
    `Matrix` and the model keeps running on the CPU, which looks like a
    successful move. Check a parameter's type afterwards, or move explicitly:

    ```julia
    using Functors: fmap
    lm_gpu = fmap(x -> x isa AbstractArray ? CuArray(x) : x, lm)
    ```

Weight-only int8 is the exception to all of this. `QuantizedInt8Matrix` is a
host-oriented leaf that dequantizes per call, so pair `quantize_int8` with CPU
execution and use fp16 for a smaller device footprint.

Importing a backend loads the matching package extension, which routes
[`Layers.sdpa`](@ref) on device arrays to [`Layers.flash_sdpa`](@ref). That is a
tiled attention kernel: it walks the key and value blocks with a running softmax
instead of materializing the full score matrix, so memory stays linear in
sequence length. It also stays NaN-safe for a fully-masked query and preserves
input precision, so fp16 in gives fp16 out. The extension is a performance path,
not a requirement for running on a device.

The tiled forward updates its accumulators in place, which reverse-mode AD cannot
trace, so `flash_sdpa` carries an explicit rule that differentiates the
materialized path instead. Training on a device therefore gets the materialized
attention backward, and only inference benefits from the tiling.

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

That file is the one GPU test outside the default suite, since it needs real
hardware. `test/gpu.jl` runs everywhere and covers the device path on JLArrays.

## Memory notes

A few allocations are sized for correctness rather than minimum footprint:

- Sliding-window models allocate `max_seq` cache slots regardless of the window,
  so a ring buffer capped at `min(max_seq, sliding_window)` would save memory.
- Tied embeddings materialize the transpose instead of sharing storage, which
  keeps the loader pure-data at a cost of roughly `vocab * hidden`.
- Mixtral batches tokens per expert but does not fuse the expert matmuls.
