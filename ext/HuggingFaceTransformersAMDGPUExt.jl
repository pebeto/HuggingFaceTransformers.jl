"""
    HuggingFaceTransformersAMDGPUExt

Loaded when both `AMDGPU` and `HuggingFaceTransformers` are imported. Routes `sdpa` on
`ROCArray`s to [`HuggingFaceTransformers.Layers.flash_sdpa`](@ref). Mirrors `HuggingFaceTransformersCUDAExt`.
"""
module HuggingFaceTransformersAMDGPUExt

using HuggingFaceTransformers
using HuggingFaceTransformers.Layers: flash_sdpa
using AMDGPU

function HuggingFaceTransformers.Layers.sdpa(
    q::ROCArray, k_t::ROCArray, v::ROCArray; scale, softcap=nothing, drop=nothing,
    sinks=nothing,
)
    drop_dev = isnothing(drop) ? nothing : ROCArray(drop)
    return flash_sdpa(q, k_t, v; scale=scale, softcap=softcap, drop=drop_dev, sinks=sinks)
end

end # module HuggingFaceTransformersAMDGPUExt
