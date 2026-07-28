"""
    GemmaConfig

Architectural hyperparameters for a Gemma2 model. Field names mirror
HuggingFace's `config.json` keys (Julianized to snake_case).

Gemma2 diverges from Llama on more axes than Mistral or Qwen, hence its
own decoder layer:

- RMSNorm uses `(1 + weight)` scaling ([`GemmaRMSNorm`](@ref)).
- The decoder layer has four RMSNorms per block (pre + post for both
  attention and MLP). See [`GemmaDecoderLayer`](@ref).
- MLP activation is GELU tanh, not SiLU ([`GeluGatedMLP`](@ref)).
- Sliding-window attention alternates per layer: even-indexed layers
  (0, 2, …) get `sliding_window`; odd-indexed get full causal.
- Attention scores and final logits are passed through
  `cap * tanh(x / cap)` softcaps (Gemma2 9B: 50 / 30).
- Query scaling uses `query_pre_attn_scalar` instead of `sqrt(head_dim)`
  (Gemma2 2B uses 224, 9B uses 256, 27B uses 144).
- `tie_word_embeddings` is always true for the Gemma2 family.
"""
Base.@kwdef struct GemmaConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    rope_theta::Float64 = 10_000.0
    rms_norm_eps::Float64 = 1.0e-6
    tie_word_embeddings::Bool = true
    sliding_window::Union{Nothing,Int} = nothing
    attn_logit_softcapping::Union{Nothing,Float64} = nothing
    final_logit_softcapping::Union{Nothing,Float64} = nothing
    query_pre_attn_scalar::Union{Nothing,Int} = nothing
end

"""
    GemmaDecoderLayer{A, N1, N2, N3, N4, M}

A Gemma2 decoder block with four RMSNorms (pre and post for both attention
and MLP) instead of Llama's pre-only pair. The forward is:

```
x = x + post_attention_layernorm(self_attn(input_layernorm(x)))
x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
```
"""
struct GemmaDecoderLayer{A,N1,N2,N3,N4,M}
    self_attn::A
    input_layernorm::N1
    post_attention_layernorm::N2
    pre_feedforward_layernorm::N3
    post_feedforward_layernorm::N4
    mlp::M
end

function (layer::GemmaDecoderLayer)(
    x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing
)
    h = layer.input_layernorm(x)
    h = layer.self_attn(h; cache=cache, step=step, position_ids=position_ids)
    h = layer.post_attention_layernorm(h)
    x = x .+ h

    h = layer.pre_feedforward_layernorm(x)
    h = layer.mlp(h)
    h = layer.post_feedforward_layernorm(h)
    return x .+ h
end

Flux.@layer GemmaDecoderLayer

"""
    GemmaModel{E, L, N}

Gemma2 transformer trunk: token embedding → stack of
`GemmaDecoderLayer`s → final `GemmaRMSNorm`. Returns hidden states; the
LM head and softcap live in `GemmaForCausalLM`.
"""
struct GemmaModel{E,L,N}
    embed_tokens::E
    layers::L
    norm::N
end

function (m::GemmaModel)(
    input_ids::AbstractMatrix{<:Integer}; caches=nothing, step=nothing, position_ids=nothing
)
    h = m.embed_tokens(input_ids)
    # Gemma scales token embeddings by sqrt(hidden_size). This is the only
    # model in the supported set that does this; it's HF parity.
    hidden_dim = size(h, 1)
    h = h .* sqrt(Float32(hidden_dim))
    for i in eachindex(m.layers)
        cache_i = isnothing(caches) ? nothing : caches[i]
        h = m.layers[i](h; cache=cache_i, step=step, position_ids=position_ids)
    end
    return m.norm(h)
end

Flux.@layer GemmaModel

"""
    GemmaForCausalLM{C, M, H}

`GemmaModel` plus a `Linear` LM head (tied to the input embedding at
load time, since `tie_word_embeddings=true` on every Gemma2 checkpoint).
When `config.final_logit_softcapping` is set, the LM head output is
passed through `cap * tanh(x / cap)` before returning.
"""
struct GemmaForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::GemmaForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    logits = lm.lm_head(h)
    cap = lm.config.final_logit_softcapping
    if !isnothing(cap)
        c = Float32(cap)
        logits = c .* tanh.(logits ./ c)
    end
    return logits
end

Flux.@layer GemmaForCausalLM
function Flux.Optimisers.trainable(lm::GemmaForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    GemmaForCausalLM(cfg::GemmaConfig)

Materialize a fresh, randomly-initialized `GemmaForCausalLM` matching
the shapes in `cfg`. The per-layer attention is built with alternating
sliding-window: even-indexed layers get `cfg.sliding_window`, odd get
plain causal.
"""
function GemmaForCausalLM(cfg::GemmaConfig)
    rope = RoPE(cfg.head_dim; base=cfg.rope_theta)
    eps = Float32(cfg.rms_norm_eps)

    query_scale = if isnothing(cfg.query_pre_attn_scalar)
        nothing
    else
        sqrt(Float32(cfg.query_pre_attn_scalar))
    end

    layers = [
        GemmaDecoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                rope;
                # Even-indexed (0-based) layers use sliding; odd use full causal.
                window_size=(i % 2 == 0) ? cfg.sliding_window : nothing,
                softcap=cfg.attn_logit_softcapping,
                query_scale=query_scale,
            ),
            GemmaRMSNorm(cfg.hidden_size, eps),
            GemmaRMSNorm(cfg.hidden_size, eps),
            GemmaRMSNorm(cfg.hidden_size, eps),
            GemmaRMSNorm(cfg.hidden_size, eps),
            GeluGatedMLP(cfg.hidden_size, cfg.intermediate_size),
        ) for i in 0:(cfg.num_hidden_layers - 1)
    ]

    model = GemmaModel(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        layers,
        GemmaRMSNorm(cfg.hidden_size, eps),
    )
    lm_head = Linear(cfg.hidden_size, cfg.vocab_size)
    return GemmaForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::GemmaForCausalLM, max_seq, batch_size; eltype=Float32)
"""
function build_caches(
    lm::GemmaForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    gemma_state_dict_map(cfg::GemmaConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal parameter table for Gemma2. The four-norm decoder
block changes the path table vs. Llama / Mistral / Qwen, so this is its
own implementation rather than a delegation.
"""
function gemma_state_dict_map(cfg::GemmaConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["model.embed_tokens.weight"] = ((:model, :embed_tokens, :weight), :transpose)

    for i in 0:(cfg.num_hidden_layers - 1)
        layer_path = (:model, :layers, i + 1)
        hf_prefix = "model.layers.$(i)"

        out["$(hf_prefix).input_layernorm.weight"] = (
            (layer_path..., :input_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).self_attn.q_proj.weight"] = (
            (layer_path..., :self_attn, :wq, :weight), :as_is
        )
        out["$(hf_prefix).self_attn.k_proj.weight"] = (
            (layer_path..., :self_attn, :wk, :weight), :as_is
        )
        out["$(hf_prefix).self_attn.v_proj.weight"] = (
            (layer_path..., :self_attn, :wv, :weight), :as_is
        )
        out["$(hf_prefix).self_attn.o_proj.weight"] = (
            (layer_path..., :self_attn, :wo, :weight), :as_is
        )
        out["$(hf_prefix).post_attention_layernorm.weight"] = (
            (layer_path..., :post_attention_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).pre_feedforward_layernorm.weight"] = (
            (layer_path..., :pre_feedforward_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).post_feedforward_layernorm.weight"] = (
            (layer_path..., :post_feedforward_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).mlp.gate_proj.weight"] = (
            (layer_path..., :mlp, :gate_proj, :weight), :as_is
        )
        out["$(hf_prefix).mlp.up_proj.weight"] = (
            (layer_path..., :mlp, :up_proj, :weight), :as_is
        )
        out["$(hf_prefix).mlp.down_proj.weight"] = (
            (layer_path..., :mlp, :down_proj, :weight), :as_is
        )
    end

    out["model.norm.weight"] = ((:model, :norm, :weight), :as_is)

    if !cfg.tie_word_embeddings
        out["lm_head.weight"] = ((:lm_head, :weight), :as_is)
    end

    return out
end

"""
    load_state_dict!(lm::GemmaForCausalLM, weights) -> lm

Populate `lm` in-place. Gemma2 always ties word embeddings, so the
`lm_head` weight is materialized from the input embedding tensor after
the rest of the state dict loads.
"""
function load_state_dict!(
    lm::GemmaForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(lm, weights, gemma_state_dict_map(lm.config))

    if lm.config.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
