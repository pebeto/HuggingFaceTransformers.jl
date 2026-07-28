"""
    HuggingFaceTransformersMetalExt

Loaded when both `Metal` and `HuggingFaceTransformers` are imported (Apple Silicon). Routes
`sdpa` on `MtlArray`s to [`HuggingFaceTransformers.Layers.flash_sdpa`](@ref). Metal has no
Float64 on-device, so run in fp32 or fp16.
"""
module HuggingFaceTransformersMetalExt

using HuggingFaceTransformers
using HuggingFaceTransformers.Layers: flash_sdpa
using Metal

function HuggingFaceTransformers.Layers.sdpa(
    q::MtlArray, k_t::MtlArray, v::MtlArray; scale, softcap=nothing, drop=nothing
)
    drop_dev = isnothing(drop) ? nothing : MtlArray(drop)
    return flash_sdpa(q, k_t, v; scale=scale, softcap=softcap, drop=drop_dev)
end

end # module HuggingFaceTransformersMetalExt
