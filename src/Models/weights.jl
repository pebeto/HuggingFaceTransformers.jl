const SAFETENSORS_FILENAME = "model.safetensors"
const SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"

"""
    load_weights(path; mmap=true) -> Dict{String, AbstractArray}

Load a HuggingFace safetensors checkpoint and return a dictionary keyed on
HF parameter names (e.g. `"model.layers.0.self_attn.q_proj.weight"`).

`path` may be:

- a directory containing `model.safetensors` (single-file checkpoint),
- a directory containing `model.safetensors.index.json` (sharded checkpoint),
- a `*.safetensors` file directly, or
- a `*.safetensors.index.json` file directly.

By default tensors are returned as lazy views over a memory map of the file, in
HF's shape: nothing is read until a tensor is used, and loading a model copies
each tensor once, straight from the file into its parameter. Pass `mmap=false`
to read everything into RAM up front instead (useful when the cache directory is
on slow storage and the tensors are about to be moved onto a GPU anyway).
"""
function load_weights(path::AbstractString; mmap::Bool=true)
    ispath(path) || throw(ArgumentError("weights path does not exist: $(path)"))

    if isdir(path)
        single = joinpath(path, SAFETENSORS_FILENAME)
        index = joinpath(path, SAFETENSORS_INDEX_FILENAME)
        if isfile(single)
            return mmap ? _mmap_safetensors(single) : load_safetensors(single; mmap=false)
        elseif isfile(index)
            return mmap ? _mmap_sharded(index) : load_sharded_safetensors(path; mmap=false)
        else
            throw(
                ArgumentError(
                    "no safetensors checkpoint in $(path): expected " *
                    "$(SAFETENSORS_FILENAME) or $(SAFETENSORS_INDEX_FILENAME)",
                ),
            )
        end
    elseif endswith(path, ".safetensors.index.json")
        return mmap ? _mmap_sharded(path) : load_sharded_safetensors(dirname(abspath(path)); mmap=false)
    elseif endswith(path, ".safetensors")
        return mmap ? _mmap_safetensors(path) : load_safetensors(path; mmap=false)
    else
        throw(
            ArgumentError(
                "unrecognized weights path: $(path) — expected a directory, " *
                "a .safetensors file, or a .safetensors.index.json file",
            ),
        )
    end
end

const _SAFETENSORS_DTYPES = Dict{String,DataType}(
    "F64" => Float64, "F32" => Float32, "F16" => Float16, "BF16" => BFloat16,
    "I64" => Int64, "I32" => Int32, "I16" => Int16, "I8" => Int8,
    "U64" => UInt64, "U32" => UInt32, "U16" => UInt16, "U8" => UInt8, "BOOL" => Bool,
)

# A safetensors file is an 8-byte header length, a JSON header, then row-major
# tensor bytes. Julia reads those bytes in reverse dimension order, so each
# tensor is a `PermutedDimsArray` over a reshaped byte view: HF's shape, no copy.
function _mmap_safetensors(file::AbstractString)
    buf = Mmap.mmap(file, Vector{UInt8})
    length(buf) >= 8 || throw(ArgumentError("not a safetensors file: $(file)"))
    header_len = Int(ltoh(reinterpret(UInt64, buf[1:8])[1]))
    header = JSON3.read(String(buf[9:(8 + header_len)]))
    data_start = 8 + header_len
    tensors = Dict{String,AbstractArray}()
    for (name, info) in pairs(header)
        name === :__metadata__ && continue
        T = get(_SAFETENSORS_DTYPES, String(info.dtype), nothing)
        isnothing(T) && throw(ArgumentError("unsupported safetensors dtype $(info.dtype) for $(name)"))
        shape = Int.(collect(info.shape))
        lo, hi = Int.(collect(info.data_offsets))
        bytes = view(buf, (data_start + lo + 1):(data_start + hi))
        raw = reshape(reinterpret(T, bytes), reverse(shape)...)
        tensors[String(name)] = PermutedDimsArray(raw, ntuple(i -> length(shape) + 1 - i, length(shape)))
    end
    return tensors
end

function _mmap_sharded(index_file::AbstractString)
    dir = dirname(abspath(index_file))
    weight_map = JSON3.read(read(index_file, String)).weight_map
    shards = Dict{String,Dict{String,AbstractArray}}()
    tensors = Dict{String,AbstractArray}()
    for (name, file) in pairs(weight_map)
        shard = get!(() -> _mmap_safetensors(joinpath(dir, String(file))), shards, String(file))
        tensors[String(name)] = shard[String(name)]
    end
    return tensors
end

"""
    _file_order(x) -> AbstractArray

A tensor's elements in the order the safetensors file stores them, which Julia
indexes with the dimensions reversed. For a lazy view from [`load_weights`](@ref)
this is the underlying memory map itself, so packed formats (MXFP4 blocks) can be
read in place; anything else is permuted into that order.
"""
_file_order(x::PermutedDimsArray) = parent(x)
_file_order(x::AbstractArray) = permutedims(x, ndims(x):-1:1)
