"""
    HuggingFaceTransformersCUDAExt

Loaded when both `CUDA` and `HuggingFaceTransformers` are imported. Routes `sdpa` on
`CuArray`s to [`HuggingFaceTransformers.Layers.flash_sdpa`](@ref). Move weights to the GPU
with Flux's `gpu(model)`; the rest of the forward already runs on CUDA.
"""
module HuggingFaceTransformersCUDAExt

using HuggingFaceTransformers
using HuggingFaceTransformers.Layers: flash_sdpa
using CUDA

function HuggingFaceTransformers.Layers.sdpa(
    q::CuArray, k_t::CuArray, v::CuArray; scale, softcap=nothing, drop=nothing
)
    # Move the host-built drop mask to the device for on-GPU slicing.
    drop_dev = isnothing(drop) ? nothing : CuArray(drop)
    return flash_sdpa(q, k_t, v; scale=scale, softcap=softcap, drop=drop_dev)
end

end # module HuggingFaceTransformersCUDAExt
