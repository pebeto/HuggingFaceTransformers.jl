"""
    GenerationConfig(; kwargs...)

Decoding defaults a checkpoint ships in `generation_config.json`, read by
[`load_generation_config`](@ref) and passed to [`generate`](@ref) as
`generation_config`. Keywords given to `generate` override these values.

Fields and defaults follow `transformers.GenerationConfig`: `do_sample=false`,
`temperature=1.0`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.0`, and empty
`eos_token_id` and `stop_strings`. `top_k`, `top_p` and `temperature` only matter
when `do_sample` is true. `max_new_tokens` and `max_length` stay `nothing` unless
the file sets them; with neither, `generate` keeps its own 16-token default
rather than HF's `max_length=20`, which counts the prompt.
"""
Base.@kwdef struct GenerationConfig
    max_new_tokens::Union{Nothing,Int} = nothing
    max_length::Union{Nothing,Int} = nothing
    do_sample::Bool = false
    temperature::Float64 = 1.0
    top_k::Union{Nothing,Int} = 50
    top_p::Float64 = 1.0
    repetition_penalty::Float64 = 1.0
    eos_token_id::Vector{Int} = Int[]
    bos_token_id::Union{Nothing,Int} = nothing
    pad_token_id::Union{Nothing,Int} = nothing
    stop_strings::Vector{String} = String[]
end

_ids(::Nothing) = Int[]
_ids(id::Integer) = [Int(id)]
_ids(ids::AbstractVector) = Int[Int(i) for i in ids]
_strings(::Nothing) = String[]
_strings(s::AbstractString) = [String(s)]
_strings(s::AbstractVector) = String[String(x) for x in s]
_maybe_int(x) = isnothing(x) ? nothing : Int(x)

function _generation_config(raw)
    get_or(key, default) = (v = get(raw, key, nothing); isnothing(v) ? default : v)
    return GenerationConfig(;
        max_new_tokens=_maybe_int(get(raw, :max_new_tokens, nothing)),
        max_length=_maybe_int(get(raw, :max_length, nothing)),
        do_sample=Bool(get_or(:do_sample, false)),
        temperature=Float64(get_or(:temperature, 1.0)),
        top_k=_maybe_int(get_or(:top_k, 50)),
        top_p=Float64(get_or(:top_p, 1.0)),
        repetition_penalty=Float64(get_or(:repetition_penalty, 1.0)),
        eos_token_id=_ids(get(raw, :eos_token_id, nothing)),
        bos_token_id=_maybe_int(get(raw, :bos_token_id, nothing)),
        pad_token_id=_maybe_int(get(raw, :pad_token_id, nothing)),
        stop_strings=_strings(get(raw, :stop_strings, nothing)),
    )
end

"""
    load_generation_config(dir) -> GenerationConfig

Read `generation_config.json` from a snapshot directory. A checkpoint without one
gets HF's fallback: the token ids from `config.json` (its `text_config` for
multimodal checkpoints) and the defaults for everything else.
"""
function load_generation_config(path::AbstractString)
    dir = isdir(path) ? path : dirname(path)
    file = joinpath(dir, "generation_config.json")
    isfile(file) && return _generation_config(JSON3.read(read(file, String)))
    model = joinpath(dir, "config.json")
    isfile(model) ||
        throw(ArgumentError("no generation_config.json or config.json in $(dir)"))
    raw = JSON3.read(read(model, String))
    haskey(raw, :eos_token_id) || !haskey(raw, :text_config) || (raw = raw[:text_config])
    ids = Dict(k => raw[k] for k in (:eos_token_id, :bos_token_id, :pad_token_id) if haskey(raw, k))
    return _generation_config(ids)
end

# The keywords a config supplies to `generate`. `n_prompt` turns HF's total
# `max_length` into a count of new tokens.
function _config_kwargs(cfg::GenerationConfig, n_prompt::Integer; batched::Bool=false)
    kw = Dict{Symbol,Any}(
        :do_sample => cfg.do_sample,
        :temperature => cfg.temperature,
        :top_k => cfg.top_k,
        :top_p => cfg.top_p,
        :repetition_penalty => cfg.repetition_penalty,
    )
    isempty(cfg.eos_token_id) || (kw[:eos_token_id] = cfg.eos_token_id)
    isempty(cfg.stop_strings) || (kw[:stop_strings] = cfg.stop_strings)
    if !isnothing(cfg.max_new_tokens)
        kw[:max_new_tokens] = cfg.max_new_tokens
    elseif !isnothing(cfg.max_length)
        kw[:max_new_tokens] = max(cfg.max_length - n_prompt, 0)
    end
    batched && !isnothing(cfg.pad_token_id) && (kw[:pad_id] = cfg.pad_token_id)
    return (; kw...)
end
_config_kwargs(::Nothing, ::Integer; _...) = (;)
