"""
    QwenConfig

Architectural hyperparameters for a Qwen2 / Qwen2.5 model. Field names
mirror HuggingFace's `config.json` keys (Julianized to snake_case).

Qwen2 is structurally a Llama variant. The two model-shape differences
carried here:

- Q/K/V projections in attention carry an additive bias (`o_proj` does
  not). `MistralConfig` and `LlamaConfig` don't expose this because their
  attention is bias-less. The `GQA` layer takes `qkv_bias=true` to wire
  this up automatically.
- `sliding_window` is honored when `use_sliding_window=true` in the
  source config; the parity loader resolves both fields and passes either
  the window size or `nothing` here.

Default `rope_theta` is `1_000_000.0` (the Qwen2 and Qwen2.5 baseline),
overridden per-checkpoint by `config.json`.
"""
Base.@kwdef struct QwenConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    rope_theta::Float64 = 1_000_000.0
    rms_norm_eps::Float64 = 1.0e-6
    tie_word_embeddings::Bool = false
    sliding_window::Union{Nothing,Int} = nothing
end

"""
    QwenForCausalLM{C, M, H}

Top-level Qwen container: a `DecoderModel` trunk wrapped in a `Linear`
LM head. Reuses the generic decoder pieces; the family-specific bits
(Q/K/V bias) live inside `GQA` via the `qkv_bias` constructor flag.
"""
struct QwenForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::QwenForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer QwenForCausalLM
function Flux.Optimisers.trainable(lm::QwenForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    QwenForCausalLM(cfg::QwenConfig)

Materialize a fresh, randomly-initialized `QwenForCausalLM` matching the
shapes in `cfg`. Weights are uninitialized for inference purposes, so they
must be replaced by a state-dict load before the model is meaningful.
"""
function QwenForCausalLM(cfg::QwenConfig)
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
                qkv_bias=true,
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
    return QwenForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::QwenForCausalLM, max_seq, batch_size; eltype=Float32)

Allocate one [`KVCache`](@ref) per decoder layer.
"""
function build_caches(
    lm::QwenForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    qwen_state_dict_map(cfg::QwenConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal parameter table for a Qwen model. Delegates to
[`_decoder_state_dict_map`](@ref) with `qkv_bias=true`,
so the table includes `q_proj.bias`, `k_proj.bias`, and `v_proj.bias`
entries that other Llama-family models don't have.
"""
function qwen_state_dict_map(cfg::QwenConfig)
    return _decoder_state_dict_map(
        cfg.num_hidden_layers, cfg.tie_word_embeddings; qkv_bias=true
    )
end

"""
    load_state_dict!(lm::QwenForCausalLM, weights) -> lm

Populate `lm` in-place from `weights`, an HF-keyed dictionary (typically
the return value of [`load_weights`](@ref)). Honors
`cfg.tie_word_embeddings` (Qwen2/2.5 0.5B and 1.5B tie; 3B and 7B don't).
"""
function load_state_dict!(
    lm::QwenForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(lm, weights, qwen_state_dict_map(lm.config))

    if lm.config.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
