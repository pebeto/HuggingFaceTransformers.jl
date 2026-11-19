"""
    load(repo_or_dir; revision, cache_dir, token, verbose, kwargs...) -> model

Load a model from the Hub, or from a local snapshot directory, in one call. The
architecture is read from `config.json`, so nothing needs naming:

```julia
lm = HuggingFaceTransformers.load("gpt2")
```

`repo_or_dir` is a repository id (`"org/name"`) or a path. Paths are used as they
are; anything else downloads through
[`HFHub.snapshot_download`](@ref HuggingFaceTransformers.HFHub.snapshot_download),
which caches into the same layout the Python library uses, so a repo already
pulled by either stack is not fetched twice.

Remaining keywords reach the model constructor. Embedding models take their
pooling that way, `load("intfloat/e5-small-v2"; pooling = :mean)` against the
`:cls` default.

The tokenizer is separate, since vision and audio checkpoints have none:

```julia
dir = HFHub.snapshot_download("gpt2")
lm, tokenizer = load(dir), Tokenizers.load_tokenizer(dir)
```

A llama.cpp GGUF file loads the same way, from a path ending in `.gguf` or from a
repository with `gguf_file` naming the file to fetch; only that file downloads.
Its weights stay quantized unless `packed=false` is passed. See
[`Models.load_gguf_model`](@ref) for the architectures it covers:

```julia
lm = load("Qwen/Qwen2.5-0.5B-Instruct-GGUF"; gguf_file="qwen2.5-0.5b-instruct-q4_k_m.gguf")
```

Unsupported architectures throw and name what was found. See
[`Models.AUTO_ARCHITECTURES`](@ref) for the supported set and
[`Models.load_pretrained`](@ref) for the offline half.
"""
function load(
    repo_or_dir::AbstractString;
    revision::AbstractString=HFHub.DEFAULT_REVISION,
    cache_dir::AbstractString=HFHub.default_cache_dir(),
    token::Union{AbstractString,Nothing}=HFHub.default_token(),
    verbose::Bool=false,
    gguf_file::Union{Nothing,AbstractString}=nothing,
    kwargs...,
)
    if isnothing(gguf_file) && isfile(repo_or_dir) && endswith(repo_or_dir, ".gguf")
        return Models.load_gguf_model(repo_or_dir; kwargs...)
    end
    if !isnothing(gguf_file)
        path = isdir(repo_or_dir) ? joinpath(repo_or_dir, gguf_file) :
            HFHub.download_file(
                repo_or_dir, gguf_file; revision=revision, cache_dir=cache_dir, token=token,
                verbose=verbose,
            )
        return Models.load_gguf_model(path; kwargs...)
    end
    dir = if isdir(repo_or_dir)
        repo_or_dir
    else
        HFHub.snapshot_download(
            repo_or_dir;
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            verbose=verbose,
        )
    end
    return Models.load_pretrained(dir; kwargs...)
end
