"""
    HuggingFaceTransformers.HFHub

Client for the HuggingFace Hub: resolves `org/repo[@revision]`, downloads
`config.json`, `tokenizer.json`, and `model.safetensors` (including sharded
variants), and caches them to `~/.cache/huggingface/hub` in the same layout
the Python `huggingface_hub` library uses, so downloads are shared across
Python and Julia tooling.
"""
module HFHub

using Downloads: Downloads
using HTTP
using JSON3

export RepoRef, parse_repo, download_file, snapshot_download, default_cache_dir

const DEFAULT_REVISION = "main"
const DEFAULT_ENDPOINT = "https://huggingface.co"

const _METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
)

"""
    default_endpoint() -> String

The Hub base URL. Overrideable via `\$HF_ENDPOINT` (matches Python).
"""
default_endpoint() = get(ENV, "HF_ENDPOINT", DEFAULT_ENDPOINT)

"""
    default_token() -> Union{String, Nothing}

Read the HuggingFace API token from `\$HF_TOKEN`, `\$HUGGING_FACE_HUB_TOKEN`,
or `~/.cache/huggingface/token` (the file the Python library writes when
`huggingface-cli login` is run). Returns `nothing` if none is set.
"""
function default_token()
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        if haskey(ENV, var) && !isempty(ENV[var])
            return strip(ENV[var])
        end
    end
    token_file = joinpath(homedir(), ".cache", "huggingface", "token")
    if isfile(token_file)
        contents = strip(read(token_file, String))
        isempty(contents) || return contents
    end
    return nothing
end

"""
    default_cache_dir() -> String

Resolve the HuggingFace hub cache directory the same way the Python
`huggingface_hub` library does: prefer `\$HF_HUB_CACHE`, then
`\$HF_HOME/hub`, then `\$XDG_CACHE_HOME/huggingface/hub`, then
`~/.cache/huggingface/hub`.
"""
function default_cache_dir()
    if haskey(ENV, "HF_HUB_CACHE")
        return ENV["HF_HUB_CACHE"]
    elseif haskey(ENV, "HF_HOME")
        return joinpath(ENV["HF_HOME"], "hub")
    elseif haskey(ENV, "XDG_CACHE_HOME")
        return joinpath(ENV["XDG_CACHE_HOME"], "huggingface", "hub")
    else
        return joinpath(homedir(), ".cache", "huggingface", "hub")
    end
end

"""
    repo_folder_name(repo_id; repo_type="model") -> String

Build the Python-compatible repo cache folder name. `meta-llama/Llama-3.2-1B`
becomes `models--meta-llama--Llama-3.2-1B`.
"""
function repo_folder_name(repo_id::AbstractString; repo_type::AbstractString="model")
    return string(repo_type, "s--", replace(repo_id, '/' => "--"))
end

"""
    RepoRef

A parsed reference to a model on the Hub. `revision` defaults to `"main"`
and may be a branch name, tag, or commit SHA.
"""
struct RepoRef
    repo_id::String
    revision::String
end

RepoRef(repo_id::AbstractString) = RepoRef(String(repo_id), DEFAULT_REVISION)

"""
    parse_repo(s; default_revision="main") -> RepoRef

Parse `"org/repo"` or `"org/repo@revision"` into a [`RepoRef`](@ref).
"""
function parse_repo(s::AbstractString; default_revision::AbstractString=DEFAULT_REVISION)
    parts = split(s, '@'; limit=2)
    repo_id = strip(parts[1])
    isempty(repo_id) && throw(ArgumentError("empty repo id in `$s`"))
    occursin('/', repo_id) ||
        throw(ArgumentError("repo id `$repo_id` must contain a `/` (org/name)"))
    revision = length(parts) == 2 ? strip(parts[2]) : default_revision
    isempty(revision) && throw(ArgumentError("empty revision in `$s`"))
    return RepoRef(String(repo_id), String(revision))
end

function _hub_url(ref::RepoRef, filename::AbstractString)
    return string(
        default_endpoint(), '/', ref.repo_id, "/resolve/", ref.revision, '/', filename
    )
end

function _auth_headers(token::Union{AbstractString,Nothing})
    return if isnothing(token)
        Pair{String,String}[]
    else
        Pair{String,String}["Authorization" => "Bearer $token"]
    end
end

"""
    FileMetadata

Result of a HEAD probe against the Hub for a single file.

* `etag`: the file's content hash (LFS sha256 or git blob sha1). Used
             as the blob filename in the cache.
* `commit`: the resolved commit SHA for the revision. Used as the
             snapshot folder name.
* `size`: file size in bytes if the Hub reported it.
"""
struct FileMetadata
    etag::String
    commit::String
    size::Union{Int,Nothing}
end

function _head_metadata(
    ref::RepoRef,
    filename::AbstractString;
    token::Union{AbstractString,Nothing}=default_token(),
    timeout::Real=10,
)
    url = _hub_url(ref, filename)
    resp = HTTP.request(
        "HEAD",
        url;
        headers=_auth_headers(token),
        redirect=false,
        status_exception=false,
        readtimeout=timeout,
    )
    if resp.status == 404
        throw(KeyError("$(ref.repo_id)@$(ref.revision):$filename"))
    elseif resp.status >= 400
        error("HEAD $url failed with HTTP $(resp.status)")
    end
    h = Dict{String,String}(lowercase(String(k)) => String(v) for (k, v) in resp.headers)
    raw_etag = get(h, "x-linked-etag", get(h, "etag", nothing))
    isnothing(raw_etag) && error("$url: no ETag returned by the Hub")
    commit = get(h, "x-repo-commit", nothing)
    isnothing(commit) && error("$url: no X-Repo-Commit returned by the Hub")
    raw_size = get(h, "x-linked-size", get(h, "content-length", nothing))
    size = isnothing(raw_size) ? nothing : tryparse(Int, raw_size)
    return FileMetadata(strip(raw_etag, '"'), commit, size)
end

function _download_to(
    url::AbstractString,
    dest::AbstractString,
    token::Union{AbstractString,Nothing};
    verbose::Bool=false,
)
    progress = verbose ? (total, now) -> _print_progress(url, total, now) : nothing
    Downloads.download(url, dest; headers=_auth_headers(token), progress=progress)
    verbose && println()
    return dest
end

function _print_progress(url, total, now)
    total == 0 && return nothing
    pct = round(100 * now / total; digits=1)
    print("\rdownloading $url  $(now)/$(total) bytes ($(pct)%)")
    return nothing
end

"""
    download_file(repo_id, filename; revision="main", cache_dir=default_cache_dir(),
                  token=default_token(), local_files_only=false, verbose=false) -> String

Download `filename` from `repo_id` at `revision` and return the absolute
path to it inside the Python-compatible cache layout:

    <cache_dir>/models--<org>--<repo>/snapshots/<commit>/<filename>

If the blob is already cached it is reused; the function only does a HEAD
request and creates the snapshot symlink. Pass `local_files_only=true` to
skip the network entirely and return the cached path (or error).
"""
function download_file(
    repo_id::AbstractString,
    filename::AbstractString;
    revision::AbstractString=DEFAULT_REVISION,
    cache_dir::AbstractString=default_cache_dir(),
    token::Union{AbstractString,Nothing}=default_token(),
    local_files_only::Bool=false,
    verbose::Bool=false,
)
    ref = RepoRef(String(repo_id), String(revision))
    repo_folder = joinpath(cache_dir, repo_folder_name(ref.repo_id))
    blobs_dir = joinpath(repo_folder, "blobs")
    refs_file = joinpath(repo_folder, "refs", ref.revision)

    if local_files_only
        isfile(refs_file) || error(
            "local_files_only=true but no cached commit for $(ref.repo_id)@$(ref.revision)",
        )
        cached_commit = strip(read(refs_file, String))
        snapshot_path = joinpath(repo_folder, "snapshots", cached_commit, filename)
        ispath(snapshot_path) ||
            error("local_files_only=true but $snapshot_path is missing from the cache")
        return snapshot_path
    end

    meta = _head_metadata(ref, filename; token=token)
    blob_path = joinpath(blobs_dir, meta.etag)
    snapshot_path = joinpath(repo_folder, "snapshots", meta.commit, filename)

    if !isfile(blob_path)
        mkpath(blobs_dir)
        tmp = tempname(blobs_dir; cleanup=false)
        try
            _download_to(_hub_url(ref, filename), tmp, token; verbose=verbose)
            if isfile(blob_path)
                rm(tmp; force=true)
            else
                mv(tmp, blob_path; force=false)
            end
        catch
            isfile(tmp) && rm(tmp; force=true)
            rethrow()
        end
    end

    mkpath(dirname(snapshot_path))
    if ispath(snapshot_path) || islink(snapshot_path)
        rm(snapshot_path; force=true)
    end
    _link_blob(blob_path, snapshot_path)

    mkpath(dirname(refs_file))
    open(refs_file, "w") do io
        return print(io, meta.commit)
    end

    return snapshot_path
end

function _link_blob(blob_path::AbstractString, snapshot_path::AbstractString)
    target = relpath(blob_path, dirname(snapshot_path))
    try
        symlink(target, snapshot_path)
    catch err
        # Windows without Developer Mode can't create symlinks unprivileged.
        if Sys.iswindows() && err isa Base.IOError
            cp(blob_path, snapshot_path; force=true)
        else
            rethrow()
        end
    end
    return snapshot_path
end

"""
    snapshot_download(repo_id; revision="main", cache_dir=default_cache_dir(),
                      token=default_token(), verbose=false) -> String

Download the files HuggingFaceTransformers needs to load a model: `config.json`,
`tokenizer.json` (and its companion JSON files), and the model weights,
whether sharded (`model.safetensors.index.json` plus each shard) or single
(`model.safetensors`).

Returns the path to the snapshot directory.
"""
function snapshot_download(
    repo_id::AbstractString;
    revision::AbstractString=DEFAULT_REVISION,
    cache_dir::AbstractString=default_cache_dir(),
    token::Union{AbstractString,Nothing}=default_token(),
    verbose::Bool=false,
)
    paths = String[]

    push!(
        paths,
        download_file(
            repo_id,
            "config.json";
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            verbose=verbose,
        ),
    )

    for fname in _METADATA_FILES
        fname == "config.json" && continue
        try
            push!(
                paths,
                download_file(
                    repo_id,
                    fname;
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    verbose=verbose,
                ),
            )
        catch err
            err isa KeyError || rethrow()
        end
    end

    sharded = try
        idx_path = download_file(
            repo_id,
            "model.safetensors.index.json";
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            verbose=verbose,
        )
        push!(paths, idx_path)
        idx = JSON3.read(read(idx_path, String))
        for shard in unique(values(idx.weight_map))
            push!(
                paths,
                download_file(
                    repo_id,
                    String(shard);
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    verbose=verbose,
                ),
            )
        end
        true
    catch err
        err isa KeyError ? false : rethrow()
    end
    if !sharded
        push!(
            paths,
            download_file(
                repo_id,
                "model.safetensors";
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                verbose=verbose,
            ),
        )
    end

    return dirname(paths[1])
end

end # module HFHub
