"""
    HuggingFaceTransformers.Generation

Decoder sampling: `generate()` (greedy / temperature / top-k / top-p /
repetition penalty / max_new_tokens / EOS), plus the minimal Jinja
subset needed to render chat templates from `tokenizer_config.json`.

`generate`'s signature mirrors HuggingFace's `model.generate` so callers
from `transformers` use it without relearning the API. Batch-1 only.
"""
module Generation

using JSON3
using Random
using ..Models:
    LlamaForCausalLM,
    MistralForCausalLM,
    QwenForCausalLM,
    GemmaForCausalLM,
    Phi3ForCausalLM,
    GPT2ForCausalLM,
    NeoXForCausalLM,
    MixtralForCausalLM,
    build_caches

const DecoderLM = Union{
    LlamaForCausalLM,
    MistralForCausalLM,
    QwenForCausalLM,
    GemmaForCausalLM,
    Phi3ForCausalLM,
    GPT2ForCausalLM,
    NeoXForCausalLM,
    MixtralForCausalLM,
}
using ..Tokenizers: Tokenizer, encode, decode, pad_token_id

export generate, speculative_generate, ChatTemplate, apply_chat_template

include("jinja/ast.jl")
include("jinja/blocks.jl")
include("jinja/lexer.jl")
include("jinja/expr_parser.jl")
include("jinja/template_parser.jl")
include("jinja/evaluator.jl")
include("jinja/renderer.jl")
include("chat_template.jl")

function _softmax(x::AbstractVector{T}) where {T<:AbstractFloat}
    m = maximum(x)
    e = exp.(x .- m)
    return e ./ sum(e)
end

function _apply_repetition_penalty!(logits::AbstractVector, ids_seen, penalty::Real)
    penalty == 1 && return logits
    penalty > 0 || throw(ArgumentError("repetition_penalty must be > 0"))
    for id in ids_seen
        idx = Int(id) + 1
        idx in axes(logits, 1) || continue
        if logits[idx] > 0
            logits[idx] /= penalty
        else
            logits[idx] *= penalty
        end
    end
    return logits
end

function _apply_temperature!(logits::AbstractVector, t::Real)
    t == 1 && return logits
    t > 0 || throw(
        ArgumentError("temperature must be > 0 (set do_sample=false for greedy decoding)"),
    )
    logits ./= t
    return logits
end

function _top_k_mask!(logits::AbstractVector, k)
    isnothing(k) && return logits
    k > 0 || throw(ArgumentError("top_k must be > 0"))
    k >= length(logits) && return logits
    threshold = sort(logits; rev=true)[k]
    for i in eachindex(logits)
        if logits[i] < threshold
            logits[i] = -Inf
        end
    end
    return logits
end

function _top_p_mask!(logits::AbstractVector, p)
    isnothing(p) && return logits
    0 < p <= 1 || throw(ArgumentError("top_p must be in (0, 1]"))
    p >= 1 && return logits
    probs = _softmax(logits)
    order = sortperm(probs; rev=true)
    cum = zero(eltype(probs))
    cutoff = length(order)
    for k in eachindex(order)
        cum += probs[order[k]]
        if cum >= p
            cutoff = k
            break
        end
    end
    for k in (cutoff + 1):length(order)
        logits[order[k]] = -Inf
    end
    return logits
end

function _sample_categorical(rng::AbstractRNG, logits::AbstractVector)
    probs = _softmax(logits)
    u = rand(rng, eltype(probs))
    cum = zero(eltype(probs))
    for i in eachindex(probs)
        cum += probs[i]
        u < cum && return i
    end
    return lastindex(probs)
end

_normalize_eos(::Nothing) = Set{Int}()
_normalize_eos(id::Integer) = Set{Int}((Int(id),))
_normalize_eos(ids::AbstractVector{<:Integer}) = Set{Int}(Int(i) for i in ids)

function _sample_step(
    logits::AbstractVector,
    ids_seen::AbstractVector{<:Integer};
    do_sample::Bool,
    temperature::Real,
    top_k,
    top_p,
    repetition_penalty::Real,
    rng::AbstractRNG,
)
    if !do_sample
        work = repetition_penalty == 1 ? logits : copy(logits)
        _apply_repetition_penalty!(work, ids_seen, repetition_penalty)
        return argmax(work) - 1
    end

    work = copy(logits)
    _apply_repetition_penalty!(work, ids_seen, repetition_penalty)
    _apply_temperature!(work, temperature)
    _top_k_mask!(work, top_k)
    _top_p_mask!(work, top_p)
    return _sample_categorical(rng, work) - 1
end

"""
    generate(lm, input_ids; kwargs...) -> Vector{Int}

Autoregressively extend `input_ids` (0-indexed HF token IDs) by up to
`max_new_tokens` tokens, returning the full `[prompt..., generated...]`
sequence. Batch-1 only.

Keyword arguments mirror HuggingFace's `model.generate`:

- `max_new_tokens::Integer = 16`: cap on tokens to add.
- `do_sample::Bool = false`: greedy when `false`; ignores
  `temperature`, `top_k`, `top_p`, and `rng`.
- `temperature::Real = 1.0`: divide logits before softmax.
- `top_k::Union{Nothing, Integer} = nothing`: keep the top-k logits.
- `top_p::Union{Nothing, Real} = nothing`: nucleus sampling cutoff in `(0, 1]`.
- `repetition_penalty::Real = 1.0`: HF rule: logits at already-seen IDs
  are divided (if positive) or multiplied (if negative) by `penalty`.
- `eos_token_id`: `Integer` or `AbstractVector{<:Integer}` that
  terminates generation when sampled.
- `rng::AbstractRNG = Random.default_rng()`.
"""
function generate(
    lm::DecoderLM,
    input_ids::AbstractVector{<:Integer};
    max_new_tokens::Integer=16,
    do_sample::Bool=false,
    temperature::Real=1.0,
    top_k=nothing,
    top_p=nothing,
    repetition_penalty::Real=1.0,
    eos_token_id=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    isempty(input_ids) && throw(ArgumentError("input_ids must be non-empty"))
    max_new_tokens >= 0 ||
        throw(ArgumentError("max_new_tokens must be ≥ 0, got $(max_new_tokens)"))

    all_ids = Int[Int(id) for id in input_ids]
    max_new_tokens == 0 && return all_ids

    n_prompt = length(input_ids)
    caches = build_caches(lm, n_prompt + max_new_tokens, 1)
    eos_set = _normalize_eos(eos_token_id)

    prefill = reshape(all_ids, :, 1)
    logits = lm(prefill; caches=caches, step=1)
    last_logits = collect(logits[:, end, 1])

    new_count = 0
    while new_count < max_new_tokens
        next_id = _sample_step(
            last_logits,
            all_ids;
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rng=rng,
        )
        push!(all_ids, next_id)
        new_count += 1

        next_id in eos_set && break
        new_count >= max_new_tokens && break

        cur = reshape([next_id], 1, 1)
        logits = lm(cur; caches=caches, step=n_prompt + new_count)
        last_logits = collect(logits[:, 1, 1])
    end

    return all_ids
end

"""
    generate(lm, prompts::AbstractVector{<:AbstractVector{<:Integer}}; kwargs...)
        -> Vector{Vector{Int}}

Generate for several prompts of differing length in one batch, returning one id
vector per prompt (prompt tokens included, as the single-prompt method does).

Prompts are padded on the **left** so every row's final prompt token lands in the
same column, which is what lets one shared decode step advance all rows together.
Attention orders by the padded indices, which is exact for rotary models because
RoPE scores depend on the difference between query and key positions, so a
per-row offset cancels. GPT-2's positions are learned embeddings instead, so an
offset would change the vector; its trunk counts each row's real tokens from the
padding mask rather than reusing the padded index. Every model in the supported
decoder set batches.

Each row stops at its own EOS while the others continue; finished rows are fed
`pad_id` and their output discarded. Greedy decoding reproduces the
single-prompt method exactly. With `do_sample`, draws are taken per row from the
shared `rng`, so the stream differs from running each prompt separately.
"""
function generate(
    lm::DecoderLM,
    prompts::AbstractVector{<:AbstractVector{<:Integer}};
    max_new_tokens::Integer=16,
    do_sample::Bool=false,
    temperature::Real=1.0,
    top_k=nothing,
    top_p=nothing,
    repetition_penalty::Real=1.0,
    eos_token_id=nothing,
    pad_id::Integer=0,
    rng::AbstractRNG=Random.default_rng(),
)
    isempty(prompts) && throw(ArgumentError("prompts must be non-empty"))
    any(isempty, prompts) && throw(ArgumentError("every prompt must be non-empty"))
    max_new_tokens >= 0 ||
        throw(ArgumentError("max_new_tokens must be ≥ 0, got $(max_new_tokens)"))

    rows = [Int[Int(id) for id in p] for p in prompts]
    max_new_tokens == 0 && return rows

    batch = length(rows)
    width = maximum(length, rows)
    eos_set = _normalize_eos(eos_token_id)

    # Left padding: each row is flushed right so column `width` holds every row's
    # last prompt token.
    ids = fill(Int(pad_id), width, batch)
    mask = falses(width, batch)
    for (b, row) in enumerate(rows)
        offset = width - length(row)
        for (i, id) in enumerate(row)
            ids[offset + i, b] = id
            mask[offset + i, b] = true
        end
    end

    caches = build_caches(lm, width + max_new_tokens, batch)
    logits = lm(
        ids;
        caches=caches,
        step=1,
        position_ids=collect(0:(width - 1)),
        padding_mask=mask,
    )
    last_logits = [collect(logits[:, end, b]) for b in 1:batch]

    finished = falses(batch)
    for t in 1:max_new_tokens
        next = fill(Int(pad_id), 1, batch)
        for b in 1:batch
            finished[b] && continue
            id = _sample_step(
                last_logits[b],
                rows[b];
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                rng=rng,
            )
            push!(rows[b], id)
            next[1, b] = id
            id in eos_set && (finished[b] = true)
        end

        (all(finished) || t == max_new_tokens) && break

        # The new column is a real position for every row: a finished row's token
        # is ignored, and rows never attend across the batch.
        mask = vcat(mask, trues(1, batch))
        logits = lm(
            next;
            caches=caches,
            step=width + t,
            position_ids=[width + t - 1],
            padding_mask=mask,
        )
        for b in 1:batch
            finished[b] || (last_logits[b] = collect(logits[:, 1, b]))
        end
    end

    return rows
end

"""
    generate(lm, tokenizer, prompt::AbstractString; kwargs...) -> String

Tokenize → generate → detokenize. The REPL-friendly entry point.
"""
function generate(lm::DecoderLM, tokenizer::Tokenizer, prompt::AbstractString; kwargs...)
    ids = encode(tokenizer, prompt)
    out_ids = generate(lm, ids; kwargs...)
    return decode(tokenizer, out_ids)
end

"""
    generate(lm, tokenizer, prompts::AbstractVector{<:AbstractString}; kwargs...)
        -> Vector{String}

Batched tokenize → generate → detokenize. `pad_id` defaults to the tokenizer's
padding token, falling back to 0 for the decoder-only checkpoints that declare
none; the padded positions are masked out either way, so the value only matters
if a row's own output is read before masking.
"""
function generate(
    lm::DecoderLM,
    tokenizer::Tokenizer,
    prompts::AbstractVector{<:AbstractString};
    pad_id::Union{Nothing,Integer}=pad_token_id(tokenizer),
    kwargs...,
)
    ids = [encode(tokenizer, p) for p in prompts]
    outs = generate(lm, ids; pad_id=isnothing(pad_id) ? 0 : pad_id, kwargs...)
    return [decode(tokenizer, o) for o in outs]
end

include("speculative.jl")

end # module Generation
