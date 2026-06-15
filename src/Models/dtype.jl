"""
    convert_eltype(model, ::Type{T}) where {T<:AbstractFloat} -> model'

Recursively walk `model` and rebuild it with every `Float*` leaf — both
arrays and scalars — converted to element type `T`. Used to switch a
loaded model between fp32, fp16, and bf16 inference.

Pure functional: returns a new model; the original is unchanged.

Use the shorthand wrappers when possible:
- [`fp16`](@ref) — convert to `Float16`.
- [`bf16`](@ref) — convert to `BFloat16`.
- [`fp32`](@ref) — convert back to `Float32`.

After conversion, allocate matching KV caches with
`build_caches(lm, max_seq, batch; eltype=T)`.

# Numerical tolerance vs the fp32 reference

Indicative per-precision error bands on last-position logits, observed
across decoder-only models in the supported set:

| dtype    | typical max element error |
|----------|---------------------------|
| `Float32`  | `< 1e-3` (the recorded reference)         |
| `BFloat16` | `1e-1` to `5e-1` (7-bit mantissa)         |
| `Float16`  | `5e-2` to `5e-1` (10-bit mantissa, narrower range) |

Deeper models accumulate more error. Per-model tolerances are
empirical: record a fixture *at the target dtype* against
`transformers` and compare against the Allspark model running at the
same dtype. The fp32 recorder scripts in `test/fixtures/` can be
adapted by passing `torch_dtype=torch.float16` (or `torch.bfloat16`).

# Known mixed-precision footnote

The current GQA forward hardcodes a few `Float32` constants — the
`-1e9` mask value, the Gemma `softcap` divisor — that cause transient
promotion to `Float32` inside attention even when weights and most
activations are lower-precision. The inference output is correct
either way, but you don't get pure-fp16 throughput. Phase 4 perf work
will widen those constants to be `eltype`-aware.
"""
function convert_eltype(model, ::Type{T}) where {T<:AbstractFloat}
    return Flux.fmap(model) do x
        if x isa AbstractArray{<:AbstractFloat}
            return T.(x)
        elseif x isa AbstractFloat
            return T(x)
        else
            return x
        end
    end
end

"""
    fp16(model) -> model'

Convert every Float leaf in `model` to `Float16`. See
[`convert_eltype`](@ref) for the full semantics and tolerance ranges.
"""
fp16(model) = convert_eltype(model, Float16)

"""
    bf16(model) -> model'

Convert every Float leaf in `model` to `BFloat16` (Brain Float, 8-bit
exponent + 7-bit mantissa). Same dynamic range as `Float32` with
reduced precision. See [`convert_eltype`](@ref).
"""
bf16(model) = convert_eltype(model, BFloat16)

"""
    fp32(model) -> model'

Convert every Float leaf in `model` back to `Float32` — the
construction default. See [`convert_eltype`](@ref).
"""
fp32(model) = convert_eltype(model, Float32)
