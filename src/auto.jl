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
    kwargs...,
)
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
