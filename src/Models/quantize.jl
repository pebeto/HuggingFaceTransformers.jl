"""
    quantize_int8(model) -> model'

Replace every `Linear` weight in `model` with a [`QuantizedInt8Matrix`](@ref)
(per-row symmetric int8), returning a new model and leaving the original
unchanged. Norms and token embeddings stay Float32. Weight-only: the matmul
dequantizes per call, trading that cost for ~4x smaller resident weights.

Expected logit error vs the fp32 reference is roughly `1e-2` to `1e-1`,
growing with depth.
"""
function quantize_int8(model)
    return fmap(model; exclude=(x -> x isa Linear || Functors.isleaf(x))) do x
        if x isa Linear && x.weight isa AbstractMatrix{<:AbstractFloat}
            return Linear(QuantizedInt8Matrix(x.weight), x.bias)
        else
            return x
        end
    end
end
