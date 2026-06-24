"""
    speculative_generate(target, draft, input_ids; kwargs...) -> Vector{Int}

Speculative decoding: the small `draft` model proposes `n_draft` tokens
autoregressively, the large `target` model verifies them in a single forward
pass, and an accept/reject rule keeps the output distributed exactly as if it
came from `target` alone (Leviathan et al. 2023; Chen et al. 2023). Returns the
full `[prompt..., generated...]` sequence of 0-indexed token IDs. Batch-1 only.

`target` and `draft` must share a vocabulary (token IDs are compared directly);
a mismatched LM-head width raises an error. Greedy (`do_sample=false`) yields
byte-for-byte the same sequence as `generate(target, …)`; sampling
(`do_sample=true`, with `temperature`) preserves the target's sampled
distribution. `top_k` / `top_p` / `repetition_penalty` are not applied here yet.

Keyword arguments:

- `n_draft::Integer = 4` — tokens drafted per round (γ).
- `max_new_tokens::Integer = 16` — cap on tokens to add.
- `do_sample::Bool = false` — greedy when `false`.
- `temperature::Real = 1.0` — divides logits before softmax (sampling only).
- `eos_token_id` — `Integer` or vector that terminates generation.
- `rng::AbstractRNG = Random.default_rng()`.

Each round sync-feeds the last committed token to both models before drafting;
that single-token pass is the scaffolding's recompute overhead (a tuned
implementation folds it into the next verify pass). The win is verifying γ
tokens per target forward instead of one; actual speedup tracks the draft's
acceptance rate.
"""
function speculative_generate(
    target::DecoderLM,
    draft::DecoderLM,
    input_ids::AbstractVector{<:Integer};
    n_draft::Integer=4,
    max_new_tokens::Integer=16,
    do_sample::Bool=false,
    temperature::Real=1.0,
    eos_token_id=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    isempty(input_ids) && throw(ArgumentError("input_ids must be non-empty"))
    n_draft >= 1 || throw(ArgumentError("n_draft must be ≥ 1, got $(n_draft)"))
    max_new_tokens >= 0 ||
        throw(ArgumentError("max_new_tokens must be ≥ 0, got $(max_new_tokens)"))

    committed = Int[Int(id) for id in input_ids]
    max_new_tokens == 0 && return committed

    n_prompt = length(committed)
    γ = Int(n_draft)
    max_seq = n_prompt + max_new_tokens + γ + 1
    tcache = build_caches(target, max_seq, 1)
    dcache = build_caches(draft, max_seq, 1)
    eos_set = _normalize_eos(eos_token_id)

    # Invariant maintained across rounds: both caches hold valid K/V for
    # positions 1..L-1, and the last committed token (position L) is unfed.
    # Prefill everything but that last token.
    if n_prompt >= 2
        pf = reshape(committed[1:(end - 1)], :, 1)
        target(pf; caches=tcache, step=1)
        draft(pf; caches=dcache, step=1)
    end

    new_count = 0
    while new_count < max_new_tokens
        L = length(committed)
        last_tok = committed[L]

        # Sync-feed the last committed token to both models: writes position L
        # and yields each model's next-token distribution for position L+1.
        p_cur = vec(target(reshape([last_tok], 1, 1); caches=tcache, step=L)[:, 1, 1])
        q_cur = vec(draft(reshape([last_tok], 1, 1); caches=dcache, step=L)[:, 1, 1])
        length(p_cur) == length(q_cur) || throw(
            DimensionMismatch(
                "target vocab ($(length(p_cur))) ≠ draft vocab ($(length(q_cur))); " *
                "speculative decoding needs a shared vocabulary",
            ),
        )

        # Draft γ tokens. Every drafted token is fed back into the draft cache
        # (including the last), so the draft stays valid to position L+γ
        # regardless of how many the target later accepts.
        draft_tokens = Vector{Int}(undef, γ)
        draft_probs = do_sample ? Vector{Vector{Float32}}(undef, γ) : nothing
        q_logits = q_cur
        for t in 1:γ
            if do_sample
                qp = _probs(q_logits, temperature)
                draft_probs[t] = qp
                draft_tokens[t] = _sample_from_probs(rng, qp) - 1
            else
                draft_tokens[t] = argmax(q_logits) - 1
            end
            q_logits = vec(
                draft(reshape([draft_tokens[t]], 1, 1); caches=dcache, step=L + t)[:, 1, 1]
            )
        end

        # Verify all γ in one target pass. Column i predicts position L+i+1, so
        # p_1 = p_cur, p_i = vlogits[:, i-1] for i ≥ 2, and the bonus
        # distribution (used only if every draft token is accepted) is the last
        # column.
        vlogits = target(reshape(draft_tokens, :, 1); caches=tcache, step=L + 1)
        target_logits(i) = i == 1 ? p_cur : vec(vlogits[:, i - 1, 1])
        bonus_logits = vec(vlogits[:, γ, 1])

        new_tokens = _verify(
            draft_tokens, draft_probs, target_logits, bonus_logits, γ;
            do_sample=do_sample, temperature=temperature, rng=rng,
        )

        for tok in new_tokens
            push!(committed, tok)
            new_count += 1
            (tok in eos_set || new_count >= max_new_tokens) && return committed
        end
    end

    return committed
end

# Accept/reject over the γ drafted tokens, returning the tokens to commit this
# round (the accepted prefix plus one correction-or-bonus token).
function _verify(
    draft_tokens, draft_probs, target_logits, bonus_logits, γ;
    do_sample::Bool, temperature::Real, rng::AbstractRNG,
)
    new_tokens = Int[]
    accepted_all = true

    for i in 1:γ
        di = draft_tokens[i]
        if !do_sample
            ti = argmax(target_logits(i)) - 1
            push!(new_tokens, ti)            # equals di on a match
            if ti != di
                accepted_all = false
                break
            end
        else
            pp = _probs(target_logits(i), temperature)
            qp = draft_probs[i]
            qd = qp[di + 1]
            ratio = qd > 0 ? pp[di + 1] / qd : Inf32
            if rand(rng, Float32) < min(1.0f0, ratio)
                push!(new_tokens, di)
            else
                residual = max.(pp .- qp, 0.0f0)
                s = sum(residual)
                corr = s > 0 ? residual ./ s : pp
                push!(new_tokens, _sample_from_probs(rng, corr) - 1)
                accepted_all = false
                break
            end
        end
    end

    if accepted_all
        bonus = if do_sample
            _sample_from_probs(rng, _probs(bonus_logits, temperature)) - 1
        else
            argmax(bonus_logits) - 1
        end
        push!(new_tokens, bonus)
    end

    return new_tokens
end

_probs(logits::AbstractVector, temperature::Real) =
    _softmax(temperature == 1 ? logits : logits ./ Float32(temperature))

function _sample_from_probs(rng::AbstractRNG, probs::AbstractVector)
    u = rand(rng, eltype(probs))
    cum = zero(eltype(probs))
    for i in eachindex(probs)
        cum += probs[i]
        u < cum && return i
    end
    return lastindex(probs)
end

"""
    speculative_generate(target, draft, tokenizer, prompt::AbstractString; kwargs...) -> String

Tokenize → speculative-generate → detokenize.
"""
function speculative_generate(
    target::DecoderLM,
    draft::DecoderLM,
    tokenizer::Tokenizer,
    prompt::AbstractString;
    kwargs...,
)
    ids = encode(tokenizer, prompt)
    out_ids = speculative_generate(target, draft, ids; kwargs...)
    return decode(tokenizer, out_ids)
end
