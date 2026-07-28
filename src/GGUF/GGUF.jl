"""
    HuggingFaceTransformers.GGUF

Reader for the GGUF container format (llama.cpp checkpoints and community
quantizations). [`load_gguf`](@ref) returns the metadata key/value table and
the tensors dequantized to `Float32`.

Tensor dtypes handled: F32, F16, Q8_0, Q4_0. The k-quants (Q4_K, Q5_K, Q6_K,
…) are not dequantized yet; loading one errors with the type name.
"""
module GGUF

export load_gguf, GGUFFile

# GGML tensor type codes (subset; see ggml.h).
const GGML_F32 = 0
const GGML_F16 = 1
const GGML_Q4_0 = 2
const GGML_Q8_0 = 8

const GGML_TYPE_NAME = Dict(
    0 => "F32",
    1 => "F16",
    2 => "Q4_0",
    3 => "Q4_1",
    6 => "Q5_0",
    7 => "Q5_1",
    8 => "Q8_0",
    9 => "Q8_1",
    10 => "Q2_K",
    11 => "Q3_K",
    12 => "Q4_K",
    13 => "Q5_K",
    14 => "Q6_K",
    15 => "Q8_K",
    28 => "BF16",
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

function _dequantize(::Val{GGML_F32}, bytes::Vector{UInt8}, n::Int)
    return Float32.(reinterpret(Float32, bytes)[1:n])
end

function _dequantize(::Val{GGML_F16}, bytes::Vector{UInt8}, n::Int)
    return Float32.(reinterpret(Float16, bytes)[1:n])
end

# Q8_0: 32-value blocks, each {Float16 d; Int8 q[32]} = 34 bytes.
function _dequantize(::Val{GGML_Q8_0}, bytes::Vector{UInt8}, n::Int)
    n % 32 == 0 || throw(ArgumentError("Q8_0 element count $(n) is not a multiple of 32"))
    out = Vector{Float32}(undef, n)
    nblocks = n ÷ 32
    pos = 1
    @inbounds for b in 1:nblocks
        d = Float32(reinterpret(Float16, bytes[pos:(pos + 1)])[1])
        pos += 2
        base = (b - 1) * 32
        for j in 1:32
            out[base + j] = d * Float32(reinterpret(Int8, bytes[pos])[1])
            pos += 1
        end
    end
    return out
end

# Q4_0: 32-value blocks, each {Float16 d; UInt8 qs[16]} = 18 bytes. Byte `k`
# holds value `k` in its low nibble and value `k+16` in its high nibble, each
# centered at 8: `x = d * (nibble - 8)`.
function _dequantize(::Val{GGML_Q4_0}, bytes::Vector{UInt8}, n::Int)
    n % 32 == 0 || throw(ArgumentError("Q4_0 element count $(n) is not a multiple of 32"))
    out = Vector{Float32}(undef, n)
    nblocks = n ÷ 32
    pos = 1
    @inbounds for b in 1:nblocks
        d = Float32(reinterpret(Float16, bytes[pos:(pos + 1)])[1])
        pos += 2
        base = (b - 1) * 32
        for k in 1:16
            byte = bytes[pos]
            pos += 1
            lo = Int(byte & 0x0f) - 8
            hi = Int(byte >> 4) - 8
            out[base + k] = d * Float32(lo)
            out[base + k + 16] = d * Float32(hi)
        end
    end
    return out
end

function _dequantize(::Val{V}, ::Vector{UInt8}, ::Int) where {V}
    name = get(GGML_TYPE_NAME, V, "type $(V)")
    throw(ArgumentError("GGUF tensor dtype $(name) is not supported yet"))
end

function _type_nbytes(t::Integer, n::Int)
    if t == GGML_F32
        4n
    elseif t == GGML_F16
        2n
    elseif t == GGML_Q8_0
        (n ÷ 32) * 34
    elseif t == GGML_Q4_0
        (n ÷ 32) * 18
    else
        throw(ArgumentError("unknown size for GGML type $(t)"))
    end
end

"""
    load_gguf(path) -> GGUFFile

Read the GGUF file at `path`, returning its metadata and the tensors
dequantized to `Float32`. Tensor dims keep GGML's contiguous-first order, which
maps directly to Julia's column-major layout.
"""
function load_gguf(path::AbstractString)
    isfile(path) || throw(ArgumentError("GGUF file not found: $(path)"))
    io = IOBuffer(read(path))

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

    # Tensor info table: (name, dims, ggml_type, offset-into-data-section).
    names = Vector{String}(undef, tensor_count)
    dims = Vector{Vector{Int}}(undef, tensor_count)
    types = Vector{Int}(undef, tensor_count)
    offsets = Vector{Int}(undef, tensor_count)
    for i in 1:tensor_count
        names[i] = _read_gguf_string(io)
        n_dims = Int(_read(io, UInt32))
        dims[i] = [Int(_read(io, UInt64)) for _ in 1:n_dims]
        types[i] = Int(_read(io, UInt32))
        offsets[i] = Int(_read(io, UInt64))
    end

    alignment = Int(get(metadata, "general.alignment", UInt32(32)))
    data_start = _align_up(position(io), alignment)
    buf = take!(io)   # full file bytes

    tensors = Dict{String,Array{Float32}}()
    tensor_types = Dict{String,String}()
    for i in 1:tensor_count
        n = prod(dims[i])
        nbytes = _type_nbytes(types[i], n)
        start = data_start + offsets[i] + 1   # 1-indexed
        raw = buf[start:(start + nbytes - 1)]
        flat = _dequantize(Val(types[i]), raw, n)
        tensors[names[i]] = reshape(flat, dims[i]...)
        tensor_types[names[i]] = get(GGML_TYPE_NAME, types[i], "type $(types[i])")
    end

    return GGUFFile(version, metadata, tensors, tensor_types)
end

end # module GGUF
