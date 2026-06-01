"""
    LlamaRopeScaling

NTK-style rotary-frequency scaling parameters used by Llama 3.x for
context-length extrapolation. Names mirror HuggingFace's `config.json`.
"""
Base.@kwdef struct LlamaRopeScaling
    factor::Float64
    low_freq_factor::Float64
    high_freq_factor::Float64
    original_max_position_embeddings::Int
end

"""
    LlamaConfig

Architectural hyperparameters for a Llama model. Field names mirror
HuggingFace's `config.json` keys (Julianized to snake_case).
"""
Base.@kwdef struct LlamaConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    rope_theta::Float64 = 500000.0
    rms_norm_eps::Float64 = 1.0e-5
    tie_word_embeddings::Bool = false
    rope_scaling::Union{Nothing,LlamaRopeScaling} = nothing
end

"""
    LlamaForCausalLM{C, M, H}

`DecoderModel` plus a `Linear` projection to the vocabulary. The `config`
field is non-trainable metadata; use it to introspect dimensions when
allocating KV caches (see [`build_caches`](@ref)).
"""
struct LlamaForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::LlamaForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer LlamaForCausalLM
Flux.Optimisers.trainable(lm::LlamaForCausalLM) = (; model=lm.model, lm_head=lm.lm_head)

"""
    LlamaForCausalLM(cfg::LlamaConfig)

Materialize a fresh, randomly-initialized `LlamaForCausalLM` matching the
shapes in `cfg`. Weights are uninitialized for inference purposes — they
must be replaced by a state-dict load before the model is meaningful.
"""
function LlamaForCausalLM(cfg::LlamaConfig)
    rope = if cfg.rope_scaling === nothing
        RoPE(cfg.head_dim; base=cfg.rope_theta)
    else
        s = cfg.rope_scaling
        RoPE(
            cfg.head_dim;
            base=cfg.rope_theta,
            scaling_factor=s.factor,
            low_freq_factor=s.low_freq_factor,
            high_freq_factor=s.high_freq_factor,
            old_context_len=s.original_max_position_embeddings,
        )
    end

    eps = Float32(cfg.rms_norm_eps)
    layers = [
        DecoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                rope,
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
    return LlamaForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::LlamaForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32)

Allocate one [`KVCache`](@ref) per decoder layer, sized for sequences up
to `max_seq` tokens and `batch_size` batch elements.
"""
function build_caches(
    lm::LlamaForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    llama_state_dict_map(cfg::LlamaConfig) -> Dict{String, Tuple{Tuple, Symbol}}

Build the HuggingFace → internal parameter table for a Llama model with
the given config. Delegates to [`Allspark.Models._decoder_state_dict_map`](@ref).
"""
function llama_state_dict_map(cfg::LlamaConfig)
    return _decoder_state_dict_map(cfg.num_hidden_layers, cfg.tie_word_embeddings)
end

"""
    load_state_dict!(lm::LlamaForCausalLM, weights) -> lm

Populate `lm` in-place from `weights`, an HF-keyed dictionary (typically
the return value of [`load_weights`](@ref)). When
`lm.config.tie_word_embeddings` is true, `lm.lm_head.weight` is
materialized from `lm.model.embed_tokens.weight` after the rest of the
state-dict is applied.
"""
function load_state_dict!(
    lm::LlamaForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(lm, weights, llama_state_dict_map(lm.config))

    if lm.config.tie_word_embeddings
        # embed.weight is (hidden, vocab); lm_head.weight is (vocab, hidden).
        # Materialize the transpose rather than share storage — the loader stays
        # pure-data this way, and the ~vocab×hidden memory cost is a Phase 4
        # optimization once we have a real workload to measure against.
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
