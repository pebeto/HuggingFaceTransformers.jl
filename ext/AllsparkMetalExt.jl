"""
    AllsparkMetalExt

Loaded when both `Metal` and `Allspark` are imported (Apple Silicon). Routes
`sdpa` on `MtlArray`s to [`Allspark.Layers.flash_sdpa`](@ref). Metal has no
Float64 on-device, so run in fp32 or fp16.
"""
module AllsparkMetalExt

using Allspark
using Allspark.Layers: flash_sdpa
using Metal

function Allspark.Layers.sdpa(
    q::MtlArray, k_t::MtlArray, v::MtlArray; scale, softcap=nothing, drop=nothing
)
    drop_dev = isnothing(drop) ? nothing : MtlArray(drop)
    return flash_sdpa(q, k_t, v; scale=scale, softcap=softcap, drop=drop_dev)
end

end # module AllsparkMetalExt
