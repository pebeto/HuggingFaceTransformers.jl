"""
    AllsparkCUDAExt

Loaded when both `CUDA` and `Allspark` are imported. Routes `sdpa` on
`CuArray`s to [`Allspark.Layers.flash_sdpa`](@ref). Move weights to the GPU
with Flux's `gpu(model)`; the rest of the forward already runs on CUDA.
"""
module AllsparkCUDAExt

using Allspark
using Allspark.Layers: flash_sdpa
using CUDA

function Allspark.Layers.sdpa(
    q::CuArray, k_t::CuArray, v::CuArray; scale, softcap=nothing, drop=nothing
)
    # Move the host-built drop mask to the device for on-GPU slicing.
    drop_dev = isnothing(drop) ? nothing : CuArray(drop)
    return flash_sdpa(q, k_t, v; scale=scale, softcap=softcap, drop=drop_dev)
end

end # module AllsparkCUDAExt
