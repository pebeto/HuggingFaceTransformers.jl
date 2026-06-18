"""
    AllsparkAMDGPUExt

Loaded when both `AMDGPU` and `Allspark` are imported. Routes `sdpa` on
`ROCArray`s to [`Allspark.Layers.flash_sdpa`](@ref). Mirrors `AllsparkCUDAExt`.
"""
module AllsparkAMDGPUExt

using Allspark
using Allspark.Layers: flash_sdpa
using AMDGPU

function Allspark.Layers.sdpa(
    q::ROCArray, k_t::ROCArray, v::ROCArray; scale, softcap=nothing, drop=nothing
)
    drop_dev = isnothing(drop) ? nothing : ROCArray(drop)
    return flash_sdpa(q, k_t, v; scale=scale, softcap=softcap, drop=drop_dev)
end

end # module AllsparkAMDGPUExt
