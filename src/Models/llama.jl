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
    TokenEmbedding{W}

Token-id → vector embedding. `weight` has shape `(hidden_dim, vocab_size)`
(column-major-friendly lookup). Input IDs are 0-indexed (HuggingFace
convention) and converted to 1-indexed internally.
"""
struct TokenEmbedding{W}
    weight::W
end

"""
    TokenEmbedding(vocab_size::Integer, hidden_dim::Integer; init = Flux.glorot_uniform)

Construct a `TokenEmbedding` with weight `(hidden_dim, vocab_size)`.
"""
function TokenEmbedding(vocab_size::Integer, hidden_dim::Integer; init=Flux.glorot_uniform)
    return TokenEmbedding(init(hidden_dim, vocab_size))
end

function (e::TokenEmbedding)(ids::AbstractVector{<:Integer})
    return e.weight[:, ids .+ 1]
end

function (e::TokenEmbedding)(ids::AbstractMatrix{<:Integer})
    seq, batch = size(ids)
    flat = e.weight[:, vec(ids) .+ 1]
    return reshape(flat, size(e.weight, 1), seq, batch)
end

Flux.@layer TokenEmbedding
Flux.Optimisers.trainable(m::TokenEmbedding) = (; weight=m.weight)

"""
    LlamaDecoderLayer{A, N1, N2, M}

One Llama transformer block: pre-norm self-attention with a residual,
followed by pre-norm SiLU-gated MLP with a residual.
"""
struct LlamaDecoderLayer{A,N1,N2,M}
    self_attn::A
    input_layernorm::N1
    post_attention_layernorm::N2
    mlp::M
end

function (layer::LlamaDecoderLayer)(
    x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing
)
    h = layer.input_layernorm(x)
    h = layer.self_attn(h; cache=cache, step=step, position_ids=position_ids)
    x = x .+ h
    h = layer.post_attention_layernorm(x)
    h = layer.mlp(h)
    return x .+ h
end

Flux.@layer LlamaDecoderLayer

"""
    LlamaModel{E, L, N}

The Llama transformer trunk: token embedding → stack of
`LlamaDecoderLayer`s → final `RMSNorm`. Returns hidden states without a
projection to vocabulary — that's the job of `LlamaForCausalLM`.
"""
struct LlamaModel{E,L,N}
    embed_tokens::E
    layers::L
    norm::N
end

function (m::LlamaModel)(
    input_ids::AbstractMatrix{<:Integer};
    caches=nothing,
    step=nothing,
    position_ids=nothing,
)
    h = m.embed_tokens(input_ids)
    for i in eachindex(m.layers)
        cache_i = isnothing(caches) ? nothing : caches[i]
        h = m.layers[i](h; cache=cache_i, step=step, position_ids=position_ids)
    end
    return m.norm(h)
end

Flux.@layer LlamaModel

"""
    LlamaForCausalLM{C, M, H}

`LlamaModel` plus a `Linear` projection to the vocabulary. The `config`
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
        LlamaDecoderLayer(
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

    model = LlamaModel(
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
        KVCache(
            cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype
        ) for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    llama_state_dict_map(cfg::LlamaConfig) -> Dict{String, Tuple{Tuple, Symbol}}

Build the HuggingFace → internal parameter table for a Llama model with
the given config. Each entry maps an HF parameter name to
`(julia_path, transform)`, ready to be fed to [`load_into!`](@ref).

Pure data: the table is generated from `cfg` only, with no dispatch on
model or layer types. Adding a new decoder-only model is a copy-paste of
this function with the per-block names swapped.

The `lm_head.weight` entry is omitted when `cfg.tie_word_embeddings` is
true — [`load_state_dict!`](@ref) materializes it from the embedding
afterwards.
"""
function llama_state_dict_map(cfg::LlamaConfig)
    return _decoder_state_dict_map(cfg.num_hidden_layers, cfg.tie_word_embeddings)
end

"""
    _decoder_state_dict_map(num_hidden_layers, tie_word_embeddings; qkv_bias=false) -> Dict

Pure-data state-dict map for any HF decoder-only LM whose parameter naming
matches the Llama / Mistral / Qwen convention (`model.embed_tokens.weight`,
`model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
`model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`,
`model.norm.weight`, optional `lm_head.weight`).

`qkv_bias=true` additionally emits entries for `q_proj.bias`, `k_proj.bias`,
and `v_proj.bias` — used by Qwen2/2.5, which biases QKV but not the output
projection.

`llama_state_dict_map`, `mistral_state_dict_map`, and `qwen_state_dict_map`
delegate here so a fix to the path table benefits every consumer.
"""
function _decoder_state_dict_map(
    num_hidden_layers::Integer, tie_word_embeddings::Bool; qkv_bias::Bool=false
)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["model.embed_tokens.weight"] =
        ((:model, :embed_tokens, :weight), :transpose)

    for i in 0:(num_hidden_layers - 1)
        layer_path = (:model, :layers, i + 1)
        hf_prefix = "model.layers.$(i)"

        out["$(hf_prefix).input_layernorm.weight"] =
            ((layer_path..., :input_layernorm, :weight), :as_is)
        out["$(hf_prefix).self_attn.q_proj.weight"] =
            ((layer_path..., :self_attn, :wq, :weight), :as_is)
        out["$(hf_prefix).self_attn.k_proj.weight"] =
            ((layer_path..., :self_attn, :wk, :weight), :as_is)
        out["$(hf_prefix).self_attn.v_proj.weight"] =
            ((layer_path..., :self_attn, :wv, :weight), :as_is)
        out["$(hf_prefix).self_attn.o_proj.weight"] =
            ((layer_path..., :self_attn, :wo, :weight), :as_is)
        if qkv_bias
            out["$(hf_prefix).self_attn.q_proj.bias"] =
                ((layer_path..., :self_attn, :wq, :bias), :as_is)
            out["$(hf_prefix).self_attn.k_proj.bias"] =
                ((layer_path..., :self_attn, :wk, :bias), :as_is)
            out["$(hf_prefix).self_attn.v_proj.bias"] =
                ((layer_path..., :self_attn, :wv, :bias), :as_is)
        end
        out["$(hf_prefix).post_attention_layernorm.weight"] =
            ((layer_path..., :post_attention_layernorm, :weight), :as_is)
        out["$(hf_prefix).mlp.gate_proj.weight"] =
            ((layer_path..., :mlp, :gate_proj, :weight), :as_is)
        out["$(hf_prefix).mlp.up_proj.weight"] =
            ((layer_path..., :mlp, :up_proj, :weight), :as_is)
        out["$(hf_prefix).mlp.down_proj.weight"] =
            ((layer_path..., :mlp, :down_proj, :weight), :as_is)
    end

    out["model.norm.weight"] = ((:model, :norm, :weight), :as_is)

    if !tie_word_embeddings
        out["lm_head.weight"] = ((:lm_head, :weight), :as_is)
    end

    return out
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
