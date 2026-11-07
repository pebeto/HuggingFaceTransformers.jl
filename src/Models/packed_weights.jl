# The bytes behind an array of 8-bit values, flat, in memory order. For a lazy
# checkpoint view this unwraps back to the memory map instead of copying.
_bytes(a::AbstractArray{UInt8}) = vec(a)
_bytes(a::AbstractArray{Float8E4M3}) = reinterpret(UInt8, vec(a))

"""
    FP8Matrix(data, scale)

An `(out, in)` weight stored as FP8 E4M3 with one `Float32` scale per output row
(compressed-tensors' channel-wise `float-quantized` format): `W[r, k] =
data[k, r] * scale[r]`. `data` is `(in, out)`, the checkpoint's byte order, so a
lazy [`load_weights`](@ref) view is used in place. Multiplying decodes on the fly
and never materializes `W`.
"""
struct FP8Matrix{D<:AbstractMatrix{Float8E4M3},S<:AbstractVector{Float32}} <: AbstractMatrix{Float32}
    data::D
    scale::S
    function FP8Matrix(data::AbstractMatrix{Float8E4M3}, scale::AbstractVector{<:Real})
        length(scale) == size(data, 2) ||
            throw(DimensionMismatch("$(length(scale)) scales for $(size(data, 2)) rows"))
        s = Vector{Float32}(scale)
        return new{typeof(data),typeof(s)}(data, s)
    end
end

Base.size(m::FP8Matrix) = (size(m.data, 2), size(m.data, 1))
Base.getindex(m::FP8Matrix, i::Int, j::Int) = Float32(m.data[j, i]) * m.scale[i]

"""
    NVFP4Matrix(packed, scales, global_scale)

An `(out, in)` weight in NVFP4 (compressed-tensors' `nvfp4-pack-quantized`): 4-bit
E2M1 values, two per byte with the low nibble first, in groups of 16 along the
input, each group scaled by an FP8 E4M3 scale divided by one `Float32` global
scale: `W[r, k] = fp4 * scales[g, r] / global_scale`. `packed` is `(in ÷ 2, out)`
and `scales` `(in ÷ 16, out)`, the checkpoint's byte order.
"""
struct NVFP4Matrix{P<:AbstractMatrix{UInt8},S<:AbstractMatrix{Float8E4M3}} <: AbstractMatrix{Float32}
    packed::P
    scales::S
    global_scale::Float32
    function NVFP4Matrix(packed::AbstractMatrix{UInt8}, scales::AbstractMatrix{Float8E4M3}, global_scale::Real)
        size(packed, 2) == size(scales, 2) && 8size(scales, 1) == size(packed, 1) || throw(
            DimensionMismatch("packed $(size(packed)) and scales $(size(scales)) disagree"),
        )
        return new{typeof(packed),typeof(scales)}(packed, scales, Float32(global_scale))
    end
end

Base.size(m::NVFP4Matrix) = (size(m.packed, 2), 2size(m.packed, 1))
function Base.getindex(m::NVFP4Matrix, i::Int, j::Int)
    lo, hi = _FP4_PAIRS[m.packed[(j - 1) ÷ 2 + 1, i] + 1]
    return (isodd(j) ? lo : hi) * (Float32(m.scales[(j - 1) ÷ 16 + 1, i]) / m.global_scale)
end

"""
    BF16Matrix(data)

A `bfloat16` matrix read as `Float32`, element by element. Wrapping a lazy
checkpoint view keeps a large embedding table in the file (half the memory of a
`Float32` copy) while lookups still produce `Float32`.
"""
struct BF16Matrix{D<:AbstractMatrix{BFloat16}} <: AbstractMatrix{Float32}
    data::D
end

Base.size(m::BF16Matrix) = size(m.data)
Base.getindex(m::BF16Matrix, i::Int, j::Int) = Float32(m.data[i, j])

# y = W x for a packed W, threaded over output rows. `row!(acc, r, x, t, n)` adds
# row r's dot products with columns t:t+n-1 of x (n ≤ 4) into acc[1:n]; reading a
# row once per four tokens is what makes a prompt affordable.
function _packed_mul!(row!, y::AbstractMatrix{Float32}, x::AbstractMatrix{Float32})
    n_tok = size(x, 2)
    Threads.@threads for r in axes(y, 1)
        acc = zeros(Float32, 4)
        for t in 1:4:n_tok
            n = min(4, n_tok - t + 1)
            fill!(acc, 0)
            row!(acc, r, x, t, n)
            @inbounds for i in 1:n
                y[r, t + i - 1] = acc[i]
            end
        end
    end
    return y
end

function Base.:*(m::FP8Matrix, x::AbstractMatrix{<:Real})
    size(x, 1) == size(m, 2) || throw(DimensionMismatch("$(size(m)) * $(size(x))"))
    xf = Matrix{Float32}(x)
    bytes, n_in, scale = _bytes(m.data), size(m.data, 1), m.scale
    return _packed_mul!(Matrix{Float32}(undef, size(m, 1), size(x, 2)), xf) do acc, r, x, t, n
        offset = (r - 1) * n_in
        if n == 1
            s = 0.0f0
            @inbounds @simd for k in 1:n_in
                s = muladd(_E4M3[bytes[offset + k] + 1], x[k, t], s)
            end
            acc[1] = s * scale[r]
        else
            t2, t3, t4 = t + min(1, n - 1), t + min(2, n - 1), t + min(3, n - 1)
            s1 = s2 = s3 = s4 = 0.0f0
            @inbounds @simd for k in 1:n_in
                w = _E4M3[bytes[offset + k] + 1]
                s1 = muladd(w, x[k, t], s1)
                s2 = muladd(w, x[k, t2], s2)
                s3 = muladd(w, x[k, t3], s3)
                s4 = muladd(w, x[k, t4], s4)
            end
            acc[1], acc[2], acc[3], acc[4] = s1 * scale[r], s2 * scale[r], s3 * scale[r], s4 * scale[r]
        end
    end
end

function Base.:*(m::NVFP4Matrix, x::AbstractMatrix{<:Real})
    size(x, 1) == size(m, 2) || throw(DimensionMismatch("$(size(m)) * $(size(x))"))
    xf = Matrix{Float32}(x)
    packed, scales = _bytes(m.packed), _bytes(m.scales)
    n_bytes, n_groups, global_scale = size(m.packed, 1), size(m.scales, 1), m.global_scale
    return _packed_mul!(Matrix{Float32}(undef, size(m, 1), size(x, 2)), xf) do acc, r, x, t, n
        t2, t3, t4 = t + min(1, n - 1), t + min(2, n - 1), t + min(3, n - 1)
        @inbounds for g in 1:n_groups
            scale = _E4M3[scales[(r - 1) * n_groups + g] + 1] / global_scale
            offset, col = (r - 1) * n_bytes + 8(g - 1), 16(g - 1)
            s1 = s2 = s3 = s4 = 0.0f0
            @simd for b in 1:8
                lo, hi = _FP4_PAIRS[packed[offset + b] + 1]
                i, j = col + 2b - 1, col + 2b
                s1 = muladd(lo, x[i, t], muladd(hi, x[j, t], s1))
                s2 = muladd(lo, x[i, t2], muladd(hi, x[j, t2], s2))
                s3 = muladd(lo, x[i, t3], muladd(hi, x[j, t3], s3))
                s4 = muladd(lo, x[i, t4], muladd(hi, x[j, t4], s4))
            end
            acc[1] = muladd(s1, scale, acc[1])
            acc[2] = muladd(s2, scale, acc[2])
            acc[3] = muladd(s3, scale, acc[3])
            acc[4] = muladd(s4, scale, acc[4])
        end
    end
end

Base.:*(m::Union{FP8Matrix,NVFP4Matrix}, x::AbstractVector{<:Real}) = vec(m * reshape(x, :, 1))

# A checkpoint's linear layer as it is stored: NVFP4 or FP8 kept packed (and, for a
# lazy view, memory-mapped), anything else copied to Float32.
function _checkpoint_linear(weights, key::AbstractString; bias::Bool=false)
    w = if haskey(weights, key * ".weight_packed")
        NVFP4Matrix(
            _file_order(weights[key * ".weight_packed"]),
            _file_order(weights[key * ".weight_scale"]),
            Float32(only(weights[key * ".weight_global_scale"])),
        )
    elseif eltype(weights[key * ".weight"]) == Float8E4M3
        FP8Matrix(_file_order(weights[key * ".weight"]), vec(weights[key * ".weight_scale"]))
    else
        _dense32(weights[key * ".weight"])
    end
    return Linear(w, bias ? _dense32(weights[key * ".bias"]) : nothing)
end

function _dense32(src::AbstractArray)
    dst = Array{Float32}(undef, size(src))
    _copy_weight!(dst, src)
    return dst
end
