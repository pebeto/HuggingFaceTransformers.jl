"""
    _resolve(obj, path::Tuple)

Walk `path` into `obj`, dispatching `Symbol` steps through `getfield` and
`Integer` steps through `getindex`. The walker is type-agnostic — any
struct + indexed-container tree works.
"""
function _resolve(obj, path::Tuple)
    for step in path
        obj = step isa Integer ? obj[step] : getfield(obj, step)
    end
    return obj
end

"""
    _apply_transform(src, transform::Symbol) -> AbstractArray

Apply a state-dict transform to an HF tensor:

- `:as_is`     — return `src` unchanged.
- `:transpose` — return `permutedims(src, (2, 1))`.

Add new transforms here when a model demands them (e.g. weight unstacking
for fused QKV checkpoints).
"""
function _apply_transform(src::AbstractArray, transform::Symbol)
    if transform === :as_is
        return src
    elseif transform === :transpose
        return permutedims(src, (2, 1))
    else
        throw(ArgumentError("unknown state-dict transform: $(transform)"))
    end
end

"""
    load_into!(target, weights, mapping) -> target

Copy weights from `weights` (an HF-keyed dictionary) into `target` in
place, driven by `mapping`. Each entry of `mapping` is
`hf_key => (path::Tuple, transform::Symbol)`, where `path` is a tuple of
`Symbol`s (field accesses) and `Integer`s (container indices) terminating
at the destination array field.

Raises `ArgumentError` listing any HF keys present in `mapping` but
missing from `weights`. Raises `DimensionMismatch` if a source array,
after the transform, doesn't fit its destination. Extra keys in
`weights` are silently ignored.
"""
function load_into!(
    target,
    weights::AbstractDict{String,<:AbstractArray},
    mapping::AbstractDict{String,<:Tuple},
)
    missing_keys = String[]
    for hf_key in keys(mapping)
        haskey(weights, hf_key) || push!(missing_keys, hf_key)
    end
    isempty(missing_keys) ||
        throw(ArgumentError("state-dict missing required keys: $(sort(missing_keys))"))

    for (hf_key, (path, transform)) in mapping
        container = _resolve(target, path[1:(end - 1)])
        dst = getfield(container, path[end])
        prepared = _apply_transform(weights[hf_key], transform)
        size(dst) == size(prepared) || throw(
            DimensionMismatch(
                "$(hf_key): target $(size(dst)) vs source $(size(prepared))",
            ),
        )
        copyto!(dst, prepared)
    end
    return target
end
