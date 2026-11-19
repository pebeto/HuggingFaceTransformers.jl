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

"""
    GGUFMatrix

An `(out, in)` weight kept in a GGUF file's own encoding, usually a block
quantization such as Q4_K, and decoded on the fly when multiplied. Each output row
is a contiguous run of blocks along the input, the order llama.cpp writes, so a
row decodes without touching any other. `bytes` is normally a view into the
memory-mapped file. Built by [`load_gguf_model`](@ref).
"""
struct GGUFMatrix{T,B<:AbstractVector{UInt8}} <: AbstractMatrix{Float32}
    bytes::B
    n_out::Int
    n_in::Int
    row_bytes::Int
end

function GGUFMatrix(type::Integer, bytes::AbstractVector{UInt8}, n_out::Integer, n_in::Integer)
    per_block, block_bytes = GGUF._block(type)
    n_in % per_block == 0 || throw(
        DimensionMismatch("rows of $(n_in) do not fill whole $(GGUF._type_name(type)) blocks"),
    )
    row_bytes = (n_in ÷ per_block) * block_bytes
    length(bytes) == n_out * row_bytes ||
        throw(DimensionMismatch("$(length(bytes)) bytes for $(n_out) rows of $(row_bytes)"))
    return GGUFMatrix{Int(type),typeof(bytes)}(bytes, Int(n_out), Int(n_in), row_bytes)
end

Base.size(m::GGUFMatrix) = (m.n_out, m.n_in)
function Base.getindex(m::GGUFMatrix{T}, i::Int, j::Int) where {T}
    per_block, block_bytes = GGUF._block(T)
    offset = (i - 1) * m.row_bytes + ((j - 1) ÷ per_block) * block_bytes
    block = GGUF._dequantize!(
        Vector{Float32}(undef, per_block), Val(T), view(m.bytes, (offset + 1):(offset + block_bytes)),
    )
    return block[(j - 1) % per_block + 1]
end

# The same rows reordered: row `i` of the result is row `rows[i]` of `m`.
function _permute_rows(m::GGUFMatrix{T}, rows::AbstractVector{<:Integer}) where {T}
    bytes = Vector{UInt8}(undef, length(m.bytes))
    for (i, r) in enumerate(rows)
        copyto!(bytes, (i - 1) * m.row_bytes + 1, m.bytes, (r - 1) * m.row_bytes + 1, m.row_bytes)
    end
    return GGUFMatrix(T, bytes, m.n_out, m.n_in)
end

"""
    GGUFEmbedding(rows::GGUFMatrix)

A token embedding table, `(hidden, vocab)`, over a GGUF `(vocab, hidden)` matrix:
looking up tokens decodes only their rows. A tied LM head multiplies by the same
`rows`.
"""
struct GGUFEmbedding{M<:GGUFMatrix} <: AbstractMatrix{Float32}
    rows::M
end

Base.size(e::GGUFEmbedding) = reverse(size(e.rows))
Base.getindex(e::GGUFEmbedding, i::Int, j::Int) = e.rows[j, i]
function Base.getindex(e::GGUFEmbedding, ::Colon, ids::AbstractVector{<:Integer})
    out = Matrix{Float32}(undef, size(e, 1), length(ids))
    for (k, id) in enumerate(ids)
        checkbounds(e, 1, id)
        _decode_row!(view(out, :, k), e.rows, e.rows.bytes, id)
    end
    return out
end

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

const _PackedMatrix = Union{FP8Matrix,NVFP4Matrix,GGUFMatrix}

# The input as the row decoders below lay a row out: as is for FP8; for NVFP4 the
# odd inputs (low nibbles) then the even ones (high nibbles).
_packed_input(::Union{FP8Matrix,GGUFMatrix}, x) = Matrix{Float32}(x)
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
_packed_bytes(m::GGUFMatrix) = m.bytes

# Scratch rows `_row_dot` needs: 16 for the fused kernels, a whole decoded row for
# the GGUF types that decode first and then multiply.
_dot_scratch(::_PackedMatrix) = 16
_dot_scratch(m::GGUFMatrix) = m.n_in
_dot_scratch(::GGUFMatrix{GGUF.GGML_Q8_0}) = 16
_dot_scratch(::GGUFMatrix{GGUF.GGML_Q4_K}) = 16
_dot_scratch(::GGUFMatrix{GGUF.GGML_Q6_K}) = 16

function _decode_row!(buf, m::GGUFMatrix{T}, bytes, r) where {T}
    offset = (r - 1) * m.row_bytes
    GGUF._dequantize!(view(buf, 1:(m.n_in)), Val(T), view(bytes, (offset + 1):(offset + m.row_bytes)))
    return buf
end

function _row_dot(m::GGUFMatrix, bytes, r, x, buf)
    row = _decode_row!(view(buf, :, 1), m, bytes, r)
    s = 0.0f0
    @inbounds @simd for k in 1:(m.n_in)
        s = muladd(row[k], x[k], s)
    end
    return s
end

# The fused kernels below cover the types that make up nearly all of a Q4_K_M,
# Q6_K or Q8_0 file: each block's integer codes meet the input before its scales
# are applied.
function _row_dot(m::GGUFMatrix{GGUF.GGML_Q8_0}, b, r, x, _)
    offset, s = (r - 1) * m.row_bytes, 0.0f0
    @inbounds for blk in 0:(m.n_in ÷ 32 - 1)
        p, y = offset + 34blk + 1, 32blk
        t = 0.0f0
        @simd for j in 1:32
            t = muladd(Float32(reinterpret(Int8, b[p + 1 + j])), x[y + j], t)
        end
        s = muladd(GGUF._f16(b, p), t, s)
    end
    return s
end

function _row_dot(m::GGUFMatrix{GGUF.GGML_Q4_K}, b, r, x, _)
    offset, s = (r - 1) * m.row_bytes, 0.0f0
    @inbounds for blk in 0:(m.n_in ÷ 256 - 1)
        p, y = offset + 144blk + 1, 256blk
        d, dmin = GGUF._f16(b, p), GGUF._f16(b, p + 2)
        for j in 0:3
            sc1, m1 = GGUF._scale_min_k4(b, p + 4, 2j)
            sc2, m2 = GGUF._scale_min_k4(b, p + 4, 2j + 1)
            q, o = p + 16 + 32j, y + 64j
            a1 = a2 = x1 = x2 = 0.0f0
            @simd for l in 0:31
                byte = b[q + l]
                xa, xb = x[o + l + 1], x[o + l + 33]
                a1 = muladd(Float32(byte & 0x0f), xa, a1)
                a2 = muladd(Float32(byte >> 4), xb, a2)
                x1 += xa
                x2 += xb
            end
            s += d * (sc1 * a1 + sc2 * a2) - dmin * (m1 * x1 + m2 * x2)
        end
    end
    return s
end

function _row_dot(m::GGUFMatrix{GGUF.GGML_Q6_K}, b, r, x, _)
    offset, s = (r - 1) * m.row_bytes, 0.0f0
    @inbounds for blk in 0:(m.n_in ÷ 256 - 1)
        p, y = offset + 210blk + 1, 256blk
        d = GGUF._f16(b, p + 208)
        for half in 0:1, g in 0:1
            ql, qh, sc = p + 64half, p + 128 + 32half, p + 192 + 8half + g
            o = y + 128half
            a1 = a2 = a3 = a4 = 0.0f0
            @simd for l in (16g):(16g + 15)
                h = b[qh + l]
                lo, hi = b[ql + l], b[ql + l + 32]
                a1 = muladd(Float32((lo & 0x0f) | ((h & 0x03) << 4)) - 32.0f0, x[o + l + 1], a1)
                a2 = muladd(Float32((hi & 0x0f) | (((h >> 2) & 0x03) << 4)) - 32.0f0, x[o + l + 33], a2)
                a3 = muladd(Float32((lo >> 4) | (((h >> 4) & 0x03) << 4)) - 32.0f0, x[o + l + 65], a3)
                a4 = muladd(Float32((hi >> 4) | (((h >> 6) & 0x03) << 4)) - 32.0f0, x[o + l + 97], a4)
            end
            i8(k) = Float32(reinterpret(Int8, b[sc + k]))
            s = muladd(d, i8(0) * a1 + i8(2) * a2 + i8(4) * a3 + i8(6) * a4, s)
        end
    end
    return s
end

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
            buf = Matrix{Float32}(undef, n_tok == 1 ? _dot_scratch(m) : n_in, 4)
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
