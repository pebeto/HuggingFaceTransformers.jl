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

# The kernels below decode with bit arithmetic instead of a table lookup, which
# keeps their loops vectorized.
#
# An E2M1 nibble, the low or high half of `b`, as a Float16 equal to 2^-14 times
# its value: the bits land where Float16 keeps its exponent and mantissa, and the
# conversion to Float32 (one instruction where the CPU has it) gets E2M1's
# subnormal 0.5 right.
@inline _fp4_lo(b::UInt16) = reinterpret(Float16, ((b << 12) & 0x8000) | ((b << 9) & 0x0e00))
@inline _fp4_hi(b::UInt16) = reinterpret(Float16, ((b << 8) & 0x8000) | ((b << 5) & 0x0e00))
const _FP4_UNIT = 2.0f0^14

# `_e4m3` reads the two NaN codes as ±480; quantized weights and scales never hold
# them.
@inline function _e4m3(b::UInt32)
    em = b & 0x7f
    normal = (em << 20) + (UInt32(120) << 23)
    subnormal = reinterpret(UInt32, Float32(em) * 0.001953125f0)
    return reinterpret(Float32, ifelse(em < 0x08, subnormal, normal) | ((b & 0x80) << 24))
end

const _PackedMatrix = Union{FP8Matrix,NVFP4Matrix}

# The input as the row decoders below lay a row out: as is for FP8; for NVFP4 the
# odd inputs (low nibbles) then the even ones (high nibbles).
_packed_input(::FP8Matrix, x) = Matrix{Float32}(x)
function _packed_input(::NVFP4Matrix, x)
    xp = Matrix{Float32}(undef, size(x))
    half = size(x, 1) ÷ 2
    @inbounds for t in axes(x, 2), b in 1:half
        xp[b, t], xp[half + b, t] = x[2b - 1, t], x[2b, t]
    end
    return xp
end

# Row r decoded into `buf`, in `_packed_input` order.
function _decode_row!(buf, m::FP8Matrix, bytes, r)
    n_in = size(m.data, 1)
    offset, scale = (r - 1) * n_in, m.scale[r]
    @inbounds @simd for k in 1:n_in
        buf[k] = _e4m3(UInt32(bytes[offset + k])) * scale
    end
    return buf
end

function _decode_row!(buf, m::NVFP4Matrix, (packed, scales), r)
    n_bytes, n_groups = size(m.packed, 1), size(m.scales, 1)
    offset, scale_offset = (r - 1) * n_bytes, (r - 1) * n_groups
    inv = _FP4_UNIT / m.global_scale
    # Two groups of 8 bytes per pass, so a pass is 16 lanes wide.
    @inbounds for p in 1:(n_groups ÷ 2)
        s1 = _e4m3(UInt32(scales[scale_offset + 2p - 1])) * inv
        s2 = _e4m3(UInt32(scales[scale_offset + 2p])) * inv
        base = 16(p - 1)
        @simd for j in 1:16
            byte, s = UInt16(packed[offset + base + j]), ifelse(j <= 8, s1, s2)
            buf[base + j] = Float32(_fp4_lo(byte)) * s
            buf[n_bytes + base + j] = Float32(_fp4_hi(byte)) * s
        end
    end
    if isodd(n_groups)
        s = _e4m3(UInt32(scales[scale_offset + n_groups])) * inv
        @inbounds for j in (n_bytes - 7):n_bytes
            byte = UInt16(packed[offset + j])
            buf[j], buf[n_bytes + j] = Float32(_fp4_lo(byte)) * s, Float32(_fp4_hi(byte)) * s
        end
    end
    return buf
end

# Row r times a single input column, decoded and summed in one pass; `acc` is
# scratch for 16 values.
function _row_dot(m::FP8Matrix, bytes, r, x, _)
    n_in = size(m.data, 1)
    offset, s = (r - 1) * n_in, 0.0f0
    @inbounds @simd for k in 1:n_in
        s = muladd(_e4m3(UInt32(bytes[offset + k])), x[k], s)
    end
    return s * m.scale[r]
end

function _row_dot(m::NVFP4Matrix, (packed, scales), r, x, acc)
    n_bytes, n_groups = size(m.packed, 1), size(m.scales, 1)
    offset, scale_offset = (r - 1) * n_bytes, (r - 1) * n_groups
    fill!(view(acc, 1:16), 0.0f0)
    @inbounds for p in 1:(n_groups ÷ 2)
        s1, s2 = _e4m3(UInt32(scales[scale_offset + 2p - 1])), _e4m3(UInt32(scales[scale_offset + 2p]))
        base = 16(p - 1)
        @simd for j in 1:16
            byte = UInt16(packed[offset + base + j])
            d = muladd(Float32(_fp4_lo(byte)), x[base + j], Float32(_fp4_hi(byte)) * x[n_bytes + base + j])
            acc[j] = muladd(ifelse(j <= 8, s1, s2), d, acc[j])
        end
    end
    s = 0.0f0
    @inbounds for j in 1:16
        s += acc[j]
    end
    if isodd(n_groups)
        d = 0.0f0
        @inbounds for j in (n_bytes - 7):n_bytes
            byte = UInt16(packed[offset + j])
            d = muladd(Float32(_fp4_lo(byte)), x[j], muladd(Float32(_fp4_hi(byte)), x[n_bytes + j], d))
        end
        s = muladd(_e4m3(UInt32(scales[scale_offset + n_groups])), d, s)
    end
    return s * _FP4_UNIT / m.global_scale
end

_packed_bytes(m::FP8Matrix) = _bytes(m.data)
_packed_bytes(m::NVFP4Matrix) = (_bytes(m.packed), _bytes(m.scales))

# Four decoded rows (the columns of `buf`) against tokens t:t+n-1 (n ≤ 4): sixteen
# sums, with every weight and input loaded once.
@inline function _dot4x4!(y, rows, buf, x, t, n)
    t1, t2, t3, t4 = t, t + min(1, n - 1), t + min(2, n - 1), t + min(3, n - 1)
    a11 = a12 = a13 = a14 = a21 = a22 = a23 = a24 = 0.0f0
    a31 = a32 = a33 = a34 = a41 = a42 = a43 = a44 = 0.0f0
    @inbounds @simd for k in axes(buf, 1)
        w1, w2, w3, w4 = buf[k, 1], buf[k, 2], buf[k, 3], buf[k, 4]
        x1, x2, x3, x4 = x[k, t1], x[k, t2], x[k, t3], x[k, t4]
        a11, a12, a13, a14 = muladd(w1, x1, a11), muladd(w1, x2, a12), muladd(w1, x3, a13), muladd(w1, x4, a14)
        a21, a22, a23, a24 = muladd(w2, x1, a21), muladd(w2, x2, a22), muladd(w2, x3, a23), muladd(w2, x4, a24)
        a31, a32, a33, a34 = muladd(w3, x1, a31), muladd(w3, x2, a32), muladd(w3, x3, a33), muladd(w3, x4, a34)
        a41, a42, a43, a44 = muladd(w4, x1, a41), muladd(w4, x2, a42), muladd(w4, x3, a43), muladd(w4, x4, a44)
    end
    sums = ((a11, a12, a13, a14), (a21, a22, a23, a24), (a31, a32, a33, a34), (a41, a42, a43, a44))
    @inbounds for i in eachindex(rows), j in 1:n
        y[rows[i], t + j - 1] = sums[i][j]
    end
    return y
end

# y = W x for a packed W, threaded over blocks of four output rows. One token is
# decoded and summed in a single pass; several decode each row once, into a buffer
# that stays in cache while it meets every token.
function Base.:*(m::_PackedMatrix, x::AbstractMatrix{<:Real})
    size(x, 1) == size(m, 2) || throw(DimensionMismatch("$(size(m)) * $(size(x))"))
    xp, data = _packed_input(m, x), _packed_bytes(m)
    (n_out, n_in), n_tok = size(m), size(x, 2)
    y = Matrix{Float32}(undef, n_out, n_tok)
    n_quads = cld(n_out, 4)
    n_tasks = min(n_quads, 2Threads.nthreads())
    @sync for c in 1:n_tasks
        Threads.@spawn begin
            buf = Matrix{Float32}(undef, n_tok == 1 ? 16 : n_in, 4)
            for q in (div((c - 1) * n_quads, n_tasks) + 1):div(c * n_quads, n_tasks)
                rows = (4q - 3):min(4q, n_out)
                if n_tok == 1
                    for r in rows
                        @inbounds y[r, 1] = _row_dot(m, data, r, xp, buf)
                    end
                    continue
                end
                for (i, r) in enumerate(rows)
                    _decode_row!(view(buf, :, i), m, data, r)
                end
                length(rows) < 4 && fill!(view(buf, :, (length(rows) + 1):4), 0.0f0)
                for t in 1:4:n_tok
                    _dot4x4!(y, rows, buf, xp, t, min(4, n_tok - t + 1))
                end
            end
        end
    end
    return y
end

Base.:*(m::_PackedMatrix, x::AbstractVector{<:Real}) = vec(m * reshape(x, :, 1))

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
