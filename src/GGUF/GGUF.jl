"""
    HuggingFaceTransformers.GGUF

Reader for the GGUF container format (llama.cpp checkpoints and community
quantizations). [`load_gguf`](@ref) returns the metadata key/value table and
the tensors dequantized to `Float32`; [`Models.load_gguf_model`](@ref
HuggingFaceTransformers.Models.load_gguf_model) builds a model from one.

Tensor dtypes handled: F32, F16, BF16, F64, the legacy block quantizations
(Q4_0, Q4_1, Q5_0, Q5_1, Q8_0), the k-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K), and
IQ4_NL, which llama.cpp substitutes for k-quants in rows too short for their
256-value blocks. The other IQ types error with the type name.
"""
module GGUF

using Mmap: Mmap

export load_gguf, GGUFFile

# GGML tensor type codes, from ggml.h.
const GGML_F32 = 0
const GGML_F16 = 1
const GGML_Q4_0 = 2
const GGML_Q4_1 = 3
const GGML_Q5_0 = 6
const GGML_Q5_1 = 7
const GGML_Q8_0 = 8
const GGML_Q2_K = 10
const GGML_Q3_K = 11
const GGML_Q4_K = 12
const GGML_Q5_K = 13
const GGML_Q6_K = 14
const GGML_IQ4_NL = 20
const GGML_F64 = 28
const GGML_BF16 = 30

const GGML_TYPE_NAME = Dict(
    0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
    8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
    14 => "Q6_K", 15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS",
    19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S", 23 => "IQ4_XS",
    24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64", 28 => "F64", 29 => "IQ1_M",
    30 => "BF16",
)

# Values per block and bytes per block for each type we decode.
const _BLOCK = Dict(
    GGML_F32 => (1, 4), GGML_F16 => (1, 2), GGML_BF16 => (1, 2), GGML_F64 => (1, 8),
    GGML_Q4_0 => (32, 18), GGML_Q4_1 => (32, 20), GGML_Q5_0 => (32, 22),
    GGML_Q5_1 => (32, 24), GGML_Q8_0 => (32, 34), GGML_Q2_K => (256, 84),
    GGML_Q3_K => (256, 110), GGML_Q4_K => (256, 144), GGML_Q5_K => (256, 176),
    GGML_Q6_K => (256, 210), GGML_IQ4_NL => (32, 18),
)

# GGUF metadata value-type codes.
const GGUF_U8, GGUF_I8 = 0, 1
const GGUF_U16, GGUF_I16 = 2, 3
const GGUF_U32, GGUF_I32 = 4, 5
const GGUF_F32 = 6
const GGUF_BOOL = 7
const GGUF_STRING = 8
const GGUF_ARRAY = 9
const GGUF_U64, GGUF_I64 = 10, 11
const GGUF_F64 = 12

"""
    GGUFFile

Parsed GGUF container. `version` is the format version, `metadata` the
key/value table, and `tensors` maps each tensor name to its `Float32` array.
`tensor_types` records each tensor's original GGML dtype name.
"""
struct GGUFFile
    version::Int
    metadata::Dict{String,Any}
    tensors::Dict{String,Array{Float32}}
    tensor_types::Dict{String,String}
end

_read(io::IO, ::Type{T}) where {T} = ltoh(read(io, T))

function _read_gguf_string(io::IO)
    len = _read(io, UInt64)
    return String(read(io, len))
end

function _read_metadata_value(io::IO, vtype::Integer)
    if vtype == GGUF_U8
        return _read(io, UInt8)
    elseif vtype == GGUF_I8
        return _read(io, Int8)
    elseif vtype == GGUF_U16
        return _read(io, UInt16)
    elseif vtype == GGUF_I16
        return _read(io, Int16)
    elseif vtype == GGUF_U32
        return _read(io, UInt32)
    elseif vtype == GGUF_I32
        return _read(io, Int32)
    elseif vtype == GGUF_F32
        return _read(io, Float32)
    elseif vtype == GGUF_BOOL
        return _read(io, UInt8) != 0
    elseif vtype == GGUF_STRING
        return _read_gguf_string(io)
    elseif vtype == GGUF_U64
        return _read(io, UInt64)
    elseif vtype == GGUF_I64
        return _read(io, Int64)
    elseif vtype == GGUF_F64
        return _read(io, Float64)
    elseif vtype == GGUF_ARRAY
        elem_type = _read(io, UInt32)
        count = _read(io, UInt64)
        return [_read_metadata_value(io, elem_type) for _ in 1:count]
    else
        throw(ArgumentError("unknown GGUF metadata value type: $(vtype)"))
    end
end

_align_up(n::Integer, a::Integer) = ((n + a - 1) ÷ a) * a

@inline _f16(b, i) = Float32(reinterpret(Float16, UInt16(b[i]) | UInt16(b[i + 1]) << 8))
@inline _u32(b, i) = UInt32(b[i]) | UInt32(b[i + 1]) << 8 | UInt32(b[i + 2]) << 16 | UInt32(b[i + 3]) << 24
@inline _i8(x) = reinterpret(Int8, x)

function _dequantize!(out, ::Val{GGML_F32}, b)
    return copyto!(out, reinterpret(Float32, b))
end
function _dequantize!(out, ::Val{GGML_F16}, b)
    return copyto!(out, reinterpret(Float16, b))
end
function _dequantize!(out, ::Val{GGML_F64}, b)
    return copyto!(out, reinterpret(Float64, b))
end
function _dequantize!(out, ::Val{GGML_BF16}, b)
    bits = reinterpret(UInt16, b)
    @inbounds for i in eachindex(out, bits)
        out[i] = reinterpret(Float32, UInt32(bits[i]) << 16)
    end
    return out
end

# Q8_0: {f16 d; i8 q[32]}, x = d * q.
function _dequantize!(out, ::Val{GGML_Q8_0}, b)
    @inbounds for blk in 0:(length(out) ÷ 32 - 1)
        p, y = 34blk + 1, 32blk
        d = _f16(b, p)
        for j in 1:32
            out[y + j] = d * _i8(b[p + 1 + j])
        end
    end
    return out
end

# Q4_0: {f16 d; u8 qs[16]}. Byte `j` holds value `j` in its low nibble and
# value `j + 16` in its high nibble, centred at 8.
function _dequantize!(out, ::Val{GGML_Q4_0}, b)
    @inbounds for blk in 0:(length(out) ÷ 32 - 1)
        p, y = 18blk + 1, 32blk
        d = _f16(b, p)
        for j in 1:16
            q = b[p + 1 + j]
            out[y + j] = d * (Int(q & 0x0f) - 8)
            out[y + j + 16] = d * (Int(q >> 4) - 8)
        end
    end
    return out
end

# IQ4_NL: Q4_0's layout, with each nibble indexing a non-uniform table.
const _IQ4_NL_VALUES = Int8[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

function _dequantize!(out, ::Val{GGML_IQ4_NL}, b)
    @inbounds for blk in 0:(length(out) ÷ 32 - 1)
        p, y = 18blk + 1, 32blk
        d = _f16(b, p)
        for j in 1:16
            q = b[p + 1 + j]
            out[y + j] = d * _IQ4_NL_VALUES[(q & 0x0f) + 1]
            out[y + j + 16] = d * _IQ4_NL_VALUES[(q >> 4) + 1]
        end
    end
    return out
end

# Q4_1: {f16 d; f16 m; u8 qs[16]}, x = d * q + m.
function _dequantize!(out, ::Val{GGML_Q4_1}, b)
    @inbounds for blk in 0:(length(out) ÷ 32 - 1)
        p, y = 20blk + 1, 32blk
        d, m = _f16(b, p), _f16(b, p + 2)
        for j in 1:16
            q = b[p + 3 + j]
            out[y + j] = muladd(d, Int(q & 0x0f), m)
            out[y + j + 16] = muladd(d, Int(q >> 4), m)
        end
    end
    return out
end

# Q5_0: {f16 d; u32 qh; u8 qs[16]}. Bit `j` of `qh` is the fifth bit of value
# `j`, bit `j + 16` that of value `j + 16`; values are centred at 16.
function _dequantize!(out, ::Val{GGML_Q5_0}, b)
    @inbounds for blk in 0:(length(out) ÷ 32 - 1)
        p, y = 22blk + 1, 32blk
        d, qh = _f16(b, p), _u32(b, p + 2)
        for j in 0:15
            q = b[p + 6 + j]
            hi0 = ((qh >> j) << 4) & 0x10
            hi1 = (qh >> (j + 12)) & 0x10
            out[y + j + 1] = d * (Int((q & 0x0f) | hi0) - 16)
            out[y + j + 17] = d * (Int((q >> 4) | hi1) - 16)
        end
    end
    return out
end

# Q5_1: {f16 d; f16 m; u32 qh; u8 qs[16]}, x = d * q + m.
function _dequantize!(out, ::Val{GGML_Q5_1}, b)
    @inbounds for blk in 0:(length(out) ÷ 32 - 1)
        p, y = 24blk + 1, 32blk
        d, m, qh = _f16(b, p), _f16(b, p + 2), _u32(b, p + 4)
        for j in 0:15
            q = b[p + 8 + j]
            hi0 = ((qh >> j) << 4) & 0x10
            hi1 = (qh >> (j + 12)) & 0x10
            out[y + j + 1] = muladd(d, Int((q & 0x0f) | hi0), m)
            out[y + j + 17] = muladd(d, Int((q >> 4) | hi1), m)
        end
    end
    return out
end

# Q2_K: {u8 scales[16]; u8 qs[64]; f16 d; f16 dmin}. Each scale byte holds a
# 4-bit scale and a 4-bit min for 16 values of 2 bits.
function _dequantize!(out, ::Val{GGML_Q2_K}, b)
    @inbounds for blk in 0:(length(out) ÷ 256 - 1)
        p, y = 84blk + 1, 256blk
        d, dmin = _f16(b, p + 80), _f16(b, p + 82)
        is = 0
        for half in 0:1
            q = p + 16 + 32half
            for shift in 0:2:6, part in 0:1
                sc = b[p + is]
                is += 1
                dl, ml = d * (sc & 0x0f), dmin * (sc >> 4)
                for l in 0:15
                    out[y += 1] = dl * ((b[q + 16part + l] >> shift) & 0x03) - ml
                end
            end
        end
    end
    return out
end

# Q3_K: {u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d}. Sixteen 6-bit signed
# scales are packed into 12 bytes; `hmask` holds each value's third bit, and a
# clear bit subtracts 4.
function _dequantize!(out, ::Val{GGML_Q3_K}, b)
    scales = Vector{Int8}(undef, 16)
    @inbounds for blk in 0:(length(out) ÷ 256 - 1)
        p, y = 110blk + 1, 256blk
        d = _f16(b, p + 108)
        a0, a1, t = _u32(b, p + 96), _u32(b, p + 100), _u32(b, p + 104)
        aux = (
            (a0 & 0x0f0f0f0f) | (((t >> 0) & 0x03030303) << 4),
            (a1 & 0x0f0f0f0f) | (((t >> 2) & 0x03030303) << 4),
            ((a0 >> 4) & 0x0f0f0f0f) | (((t >> 4) & 0x03030303) << 4),
            ((a1 >> 4) & 0x0f0f0f0f) | (((t >> 6) & 0x03030303) << 4),
        )
        for k in 0:15
            scales[k + 1] = _i8(UInt8((aux[k ÷ 4 + 1] >> (8 * (k % 4))) & 0xff))
        end
        is, m = 1, 0x01
        for half in 0:1
            q = p + 32 + 32half
            for shift in 0:2:6
                for part in 0:1
                    dl = d * (scales[is] - 32)
                    is += 1
                    for l in 0:15
                        i = 16part + l
                        v = Int((b[q + i] >> shift) & 0x03) - (b[p + i] & m == 0 ? 4 : 0)
                        out[y += 1] = dl * v
                    end
                end
                m <<= 1
            end
        end
    end
    return out
end

# The 6-bit scale and min of sub-block `j` in the 12 scale bytes at `s` that
# Q4_K and Q5_K share.
@inline function _scale_min_k4(b, s, j)
    if j < 4
        return b[s + j] & 0x3f, b[s + j + 4] & 0x3f
    end
    sc = (b[s + j + 4] & 0x0f) | ((b[s + j - 4] >> 6) << 4)
    mn = (b[s + j + 4] >> 4) | ((b[s + j] >> 6) << 4)
    return sc, mn
end

# Q4_K: {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}, eight sub-blocks of 32.
function _dequantize!(out, ::Val{GGML_Q4_K}, b)
    @inbounds for blk in 0:(length(out) ÷ 256 - 1)
        p, y = 144blk + 1, 256blk
        d, dmin = _f16(b, p), _f16(b, p + 2)
        for j in 0:3
            sc1, m1 = _scale_min_k4(b, p + 4, 2j)
            sc2, m2 = _scale_min_k4(b, p + 4, 2j + 1)
            d1, n1, d2, n2 = d * sc1, dmin * m1, d * sc2, dmin * m2
            q = p + 16 + 32j
            for l in 0:31
                out[y + 64j + l + 1] = d1 * (b[q + l] & 0x0f) - n1
                out[y + 64j + l + 33] = d2 * (b[q + l] >> 4) - n2
            end
        end
    end
    return out
end

# Q5_K: {f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]}.
function _dequantize!(out, ::Val{GGML_Q5_K}, b)
    @inbounds for blk in 0:(length(out) ÷ 256 - 1)
        p, y = 176blk + 1, 256blk
        d, dmin = _f16(b, p), _f16(b, p + 2)
        for j in 0:3
            sc1, m1 = _scale_min_k4(b, p + 4, 2j)
            sc2, m2 = _scale_min_k4(b, p + 4, 2j + 1)
            d1, n1, d2, n2 = d * sc1, dmin * m1, d * sc2, dmin * m2
            u1, u2 = 0x01 << 2j, 0x02 << 2j
            q = p + 48 + 32j
            for l in 0:31
                h = b[p + 16 + l]
                out[y + 64j + l + 1] = d1 * ((b[q + l] & 0x0f) + (h & u1 != 0 ? 16 : 0)) - n1
                out[y + 64j + l + 33] = d2 * ((b[q + l] >> 4) + (h & u2 != 0 ? 16 : 0)) - n2
            end
        end
    end
    return out
end

# Q6_K: {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d}. Low 4 bits in `ql`, high
# 2 bits in `qh`, centred at 32, one signed 8-bit scale per 16 values.
function _dequantize!(out, ::Val{GGML_Q6_K}, b)
    @inbounds for blk in 0:(length(out) ÷ 256 - 1)
        p, y = 210blk + 1, 256blk
        d = _f16(b, p + 208)
        for half in 0:1
            ql, qh, sc = p + 64half, p + 128 + 32half, p + 192 + 8half
            o = y + 128half
            for l in 0:31
                is = l ÷ 16
                h = b[qh + l]
                q1 = Int((b[ql + l] & 0x0f) | ((h & 0x03) << 4)) - 32
                q2 = Int((b[ql + l + 32] & 0x0f) | (((h >> 2) & 0x03) << 4)) - 32
                q3 = Int((b[ql + l] >> 4) | (((h >> 4) & 0x03) << 4)) - 32
                q4 = Int((b[ql + l + 32] >> 4) | (((h >> 6) & 0x03) << 4)) - 32
                out[o + l + 1] = d * _i8(b[sc + is]) * q1
                out[o + l + 33] = d * _i8(b[sc + is + 2]) * q2
                out[o + l + 65] = d * _i8(b[sc + is + 4]) * q3
                out[o + l + 97] = d * _i8(b[sc + is + 6]) * q4
            end
        end
    end
    return out
end

_type_name(t) = get(GGML_TYPE_NAME, t, "type $(t)")

function _block(t::Integer)
    haskey(_BLOCK, t) ||
        throw(ArgumentError("GGUF tensor dtype $(_type_name(t)) is not supported yet"))
    return _BLOCK[t]
end

struct _TensorInfo
    dims::Vector{Int}
    type::Int
    offset::Int
end

# A GGUF file read up to its tensor data, which stays memory-mapped so tensors
# can be decoded one at a time.
struct _GGUFIndex
    version::Int
    metadata::Dict{String,Any}
    tensors::Dict{String,_TensorInfo}
    buf::Vector{UInt8}
end

function _open_gguf(path::AbstractString)
    isfile(path) || throw(ArgumentError("GGUF file not found: $(path)"))
    buf = Mmap.mmap(path, Vector{UInt8})
    io = IOBuffer(buf)

    magic = read(io, 4)
    magic == b"GGUF" || throw(ArgumentError("not a GGUF file (bad magic): $(path)"))
    version = Int(_read(io, UInt32))
    tensor_count = Int(_read(io, UInt64))
    kv_count = Int(_read(io, UInt64))

    metadata = Dict{String,Any}()
    for _ in 1:kv_count
        key = _read_gguf_string(io)
        vtype = _read(io, UInt32)
        metadata[key] = _read_metadata_value(io, vtype)
    end

    entries = Vector{Tuple{String,Vector{Int},Int,Int}}(undef, tensor_count)
    for i in 1:tensor_count
        name = _read_gguf_string(io)
        n_dims = Int(_read(io, UInt32))
        dims = [Int(_read(io, UInt64)) for _ in 1:n_dims]
        entries[i] = (name, dims, Int(_read(io, UInt32)), Int(_read(io, UInt64)))
    end
    alignment = Int(get(metadata, "general.alignment", UInt32(32)))
    data_start = _align_up(position(io), alignment)
    tensors = Dict(name => _TensorInfo(dims, t, data_start + offset) for (name, dims, t, offset) in entries)
    return _GGUFIndex(version, metadata, tensors, buf)
end

# Decode one tensor to `Float32`, in GGML's contiguous-first dimension order.
function _tensor(idx::_GGUFIndex, name::AbstractString)
    info = idx.tensors[name]
    n = prod(info.dims; init=1)
    per_block, block_bytes = _block(info.type)
    n % per_block == 0 || throw(
        ArgumentError("$(name): $(n) values do not fill whole $(_type_name(info.type)) blocks"),
    )
    nbytes = (n ÷ per_block) * block_bytes
    bytes = view(idx.buf, (info.offset + 1):(info.offset + nbytes))
    out = Vector{Float32}(undef, n)
    # Blocks decode independently, so large tensors split across threads in
    # runs of whole blocks.
    blocks = n ÷ per_block
    chunks = min(Threads.nthreads(), cld(n, 1 << 18))
    Threads.@threads for c in 1:chunks
        lo, hi = div((c - 1) * blocks, chunks), div(c * blocks, chunks)
        _dequantize!(
            view(out, (lo * per_block + 1):(hi * per_block)),
            Val(info.type),
            view(bytes, (lo * block_bytes + 1):(hi * block_bytes)),
        )
    end
    return reshape(out, info.dims...)
end

"""
    load_gguf(path) -> GGUFFile

Read the GGUF file at `path`, returning its metadata and the tensors
dequantized to `Float32`. Tensor dims keep GGML's contiguous-first order, which
maps directly to Julia's column-major layout.
"""
function load_gguf(path::AbstractString)
    idx = _open_gguf(path)
    tensors = Dict{String,Array{Float32}}(name => _tensor(idx, name) for name in keys(idx.tensors))
    types = Dict(name => _type_name(info.type) for (name, info) in idx.tensors)
    return GGUFFile(idx.version, idx.metadata, tensors, types)
end

end # module GGUF
