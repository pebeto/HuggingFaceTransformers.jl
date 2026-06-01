"""
    MistralConfig

Architectural hyperparameters for a Mistral model. Field names mirror
HuggingFace's `config.json` keys (Julianized to snake_case).

Mistral is structurally a Llama variant: same RMSNorm, SwiGLU MLP, GQA
attention, RoPE. The two model-shape differences carried in this struct:

- `sliding_window`: when set, attention is restricted to the most-recent
  `sliding_window` positions (inclusive). `nothing` is full causal.
- No `rope_scaling`: Mistral uses plain RoPE; the default `rope_theta`
  is `10_000.0` for v0.1 / v0.2 and `1_000_000.0` for v0.3 / Nemo, both
  configured per-checkpoint via `config.json`.
"""
Base.@kwdef struct MistralConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    rope_theta::Float64 = 10_000.0
    rms_norm_eps::Float64 = 1.0e-5
    tie_word_embeddings::Bool = false
    sliding_window::Union{Nothing,Int} = nothing
end

"""
    MistralForCausalLM{C, M, H}

Top-level Mistral container: a `DecoderModel` trunk (reused as the generic
decoder-only transformer body) wrapped in a `Linear` LM head. The
`config` field is non-trainable metadata; `build_caches` reads it to size
KV caches.
"""
struct MistralForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::MistralForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer MistralForCausalLM
function Flux.Optimisers.trainable(lm::MistralForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    MistralForCausalLM(cfg::MistralConfig)

Materialize a fresh, randomly-initialized `MistralForCausalLM` matching the
shapes in `cfg`. Weights are uninitialized for inference purposes — they
must be replaced by a state-dict load before the model is meaningful.
"""
function MistralForCausalLM(cfg::MistralConfig)
    rope = RoPE(cfg.head_dim; base=cfg.rope_theta)
    eps = Float32(cfg.rms_norm_eps)

    layers = [
        DecoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                rope;
                window_size=cfg.sliding_window,
            ),
            RMSNorm(cfg.hidden_size, eps),
            RMSNorm(cfg.hidden_size, eps),
            SiLUGatedMLP(cfg.hidden_size, cfg.intermediate_size),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    model = DecoderModel(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        layers,
        RMSNorm(cfg.hidden_size, eps),
    )
    lm_head = Linear(cfg.hidden_size, cfg.vocab_size)
    return MistralForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::MistralForCausalLM, max_seq, batch_size; eltype=Float32)

Allocate one [`KVCache`](@ref) per decoder layer. The cache is sized to
`max_seq` slots regardless of `sliding_window` — Phase 4 will introduce a
ring-buffer cache to cap memory at `min(max_seq, sliding_window)`.
"""
function build_caches(
    lm::MistralForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    mistral_state_dict_map(cfg::MistralConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal parameter table for a Mistral model. The HF
parameter naming is identical to Llama's, so this delegates to
[`Allspark.Models._decoder_state_dict_map`](@ref).
"""
function mistral_state_dict_map(cfg::MistralConfig)
    return _decoder_state_dict_map(cfg.num_hidden_layers, cfg.tie_word_embeddings)
end

"""
    load_state_dict!(lm::MistralForCausalLM, weights) -> lm

Populate `lm` in-place from `weights`, an HF-keyed dictionary (typically
the return value of [`load_weights`](@ref)). Tied embeddings are
materialized from the input embedding the same way as for Llama.
"""
function load_state_dict!(
    lm::MistralForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(lm, weights, mistral_state_dict_map(lm.config))

    if lm.config.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
