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

By default tensors are returned as mmap-backed views; pass `mmap=false` to
force an eager copy into RAM (useful when the cache directory is on slow
storage and the tensors are about to be moved onto a GPU anyway).
"""
function load_weights(path::AbstractString; mmap::Bool=true)
    ispath(path) || throw(ArgumentError("weights path does not exist: $(path)"))

    if isdir(path)
        single = joinpath(path, SAFETENSORS_FILENAME)
        index = joinpath(path, SAFETENSORS_INDEX_FILENAME)
        if isfile(single)
            return load_safetensors(single; mmap=mmap)
        elseif isfile(index)
            return load_sharded_safetensors(path; mmap=mmap)
        else
            throw(
                ArgumentError(
                    "no safetensors checkpoint in $(path): expected " *
                    "$(SAFETENSORS_FILENAME) or $(SAFETENSORS_INDEX_FILENAME)",
                ),
            )
        end
    elseif endswith(path, ".safetensors.index.json")
        return load_sharded_safetensors(dirname(abspath(path)); mmap=mmap)
    elseif endswith(path, ".safetensors")
        return load_safetensors(path; mmap=mmap)
    else
        throw(
            ArgumentError(
                "unrecognized weights path: $(path) — expected a directory, " *
                "a .safetensors file, or a .safetensors.index.json file",
            ),
        )
    end
end
