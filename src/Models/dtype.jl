"""
    convert_eltype(model, ::Type{T}) where {T<:AbstractFloat} -> model'

Rebuild `model` with every `Float` leaf (arrays and scalars) converted to
`T`, returning a new model and leaving the original unchanged. Use the
[`fp16`](@ref) / [`bf16`](@ref) / [`fp32`](@ref) wrappers for the common
targets, and `build_caches(lm, max_seq, batch; eltype=T)` for matching caches.

Per-dtype error on last-position logits runs roughly `<1e-3` (Float32),
`5e-2` to `5e-1` (Float16), `1e-1` to `5e-1` (BFloat16), growing with depth.
GQA's `-1e9` mask and softcap constants stay `Float32`, so attention
promotes transiently even in low precision.
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

Convert every Float leaf in `model` to `BFloat16`. See [`convert_eltype`](@ref).
Intended for the GPU path: on Julia 1.12.6 `BFloat16` deadlocks in CPU LLVM
codegen, so prefer [`fp16`](@ref) for CPU low-precision runs.
"""
bf16(model) = convert_eltype(model, BFloat16)

"""
    fp32(model) -> model'

Convert every Float leaf in `model` back to `Float32`, the
construction default. See [`convert_eltype`](@ref).
"""
fp32(model) = convert_eltype(model, Float32)
