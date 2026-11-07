"""
    Qwen3_5Config

Architectural hyperparameters for the text side of Qwen3.5 (`qwen3_5`: Qwen3.5,
Qwen3.6, Qwen3.8; the vision tower and multi-token-prediction head are not
loaded). Field names mirror HuggingFace's `text_config` keys.

The model interleaves two kinds of block, as `layer_types` lists them:

- `:linear_attention`: [`GatedDeltaNet`](@ref) linear attention, with
  `linear_num_key_heads` / `linear_num_value_heads` heads of
  `linear_key_head_dim` / `linear_value_head_dim` and a causal convolution of
  `linear_conv_kernel_dim`.
- `:full_attention`: grouped-query attention with a sigmoid output gate
  (`attn_output_gate`), zero-centered query/key norms, and RoPE on the first
  `partial_rotary_factor` of each head.

Every RMSNorm outside the delta net scales by `1 + weight`. `hf_prefix` is where
the text model's weights live: `model.language_model` in the multimodal
checkpoints Qwen publishes, `model` in a text-only one.
"""
Base.@kwdef struct Qwen3_5Config
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    linear_num_key_heads::Int
    linear_num_value_heads::Int
    linear_key_head_dim::Int
    linear_value_head_dim::Int
    linear_conv_kernel_dim::Int = 4
    max_position_embeddings::Int
    rope_theta::Float64 = 10_000_000.0
    partial_rotary_factor::Float64 = 0.25
    rms_norm_eps::Float64 = 1.0e-6
    attn_output_gate::Bool = true
    attention_bias::Bool = false
    tie_word_embeddings::Bool = false
    # HF's default: every fourth layer is full attention.
    layer_types::Vector{Symbol} = [
        i % 4 == 0 ? :full_attention : :linear_attention for i in 1:num_hidden_layers
    ]
    hf_prefix::String = "model.language_model"
end

"""
    Qwen3_5ForCausalLM

Qwen3.5's language model: a [`DecoderModel`](@ref) whose layers alternate
[`GatedDeltaNet`](@ref) and gated attention blocks, plus the LM head. Build caches
with [`build_caches`](@ref): attention layers get a [`KVCache`](@ref) and delta-net
layers a [`DeltaNetCache`](@ref).
"""
struct Qwen3_5ForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

Flux.@layer Qwen3_5ForCausalLM
function Flux.Optimisers.trainable(lm::Qwen3_5ForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

function (lm::Qwen3_5ForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    return lm.lm_head(lm.model(input_ids; kwargs...))
end

function _qwen3_5_mixer(cfg::Qwen3_5Config, rope, kind::Symbol)
    if kind === :full_attention
        return GQA(
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, rope;
            qkv_bias=cfg.attention_bias, wo_bias=cfg.attention_bias,
            qk_norm_eps=cfg.rms_norm_eps, qk_norm_zero_centered=true,
            output_gate=cfg.attn_output_gate,
        )
    elseif kind === :linear_attention
        return GatedDeltaNet(
            cfg.hidden_size, cfg.linear_num_key_heads, cfg.linear_num_value_heads,
            cfg.linear_key_head_dim, cfg.linear_value_head_dim;
            conv_kernel=cfg.linear_conv_kernel_dim, eps=cfg.rms_norm_eps,
        )
    end
    throw(ArgumentError("unsupported Qwen3.5 layer type `$(kind)`"))
end

function Qwen3_5ForCausalLM(cfg::Qwen3_5Config)
    length(cfg.layer_types) == cfg.num_hidden_layers || throw(
        ArgumentError(
            "layer_types lists $(length(cfg.layer_types)) layers for $(cfg.num_hidden_layers)",
        ),
    )
    rope = RoPE(
        cfg.head_dim; base=cfg.rope_theta,
        rotary_dim=round(Int, cfg.head_dim * cfg.partial_rotary_factor),
    )
    eps = Float32(cfg.rms_norm_eps)
    layers = [
        DecoderLayer(
            _qwen3_5_mixer(cfg, rope, kind),
            GemmaRMSNorm(cfg.hidden_size, eps),
            GemmaRMSNorm(cfg.hidden_size, eps),
            SiLUGatedMLP(cfg.hidden_size, cfg.intermediate_size),
        ) for kind in cfg.layer_types
    ]
    model = DecoderModel(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size), layers, GemmaRMSNorm(cfg.hidden_size, eps)
    )
    return Qwen3_5ForCausalLM(cfg, model, Linear(cfg.hidden_size, cfg.vocab_size))
end

"""
    Qwen3_5ForCausalLM(cfg, weights)

Build the model straight from an HF-keyed weight dictionary, as
[`load_weights`](@ref) returns it, without first allocating dense parameters.
Linear layers stored as NVFP4 or FP8 (compressed-tensors checkpoints such as
`unsloth/Qwen3.8-27B-NVFP4`) stay packed, as [`NVFP4Matrix`](@ref) and
[`FP8Matrix`](@ref), and bf16 embeddings stay in the file behind a
[`BF16Matrix`](@ref); everything else is copied to `Float32`. Weights are decoded
exactly while multiplying, and activations stay in `Float32`.
"""
function Qwen3_5ForCausalLM(cfg::Qwen3_5Config, weights::AbstractDict{String,<:AbstractArray})
    P = cfg.hf_prefix
    eps = Float32(cfg.rms_norm_eps)
    vector(key) = vec(_dense32(weights[key]))
    norm(key) = GemmaRMSNorm(vector(key), eps)
    rope = RoPE(
        cfg.head_dim; base=cfg.rope_theta,
        rotary_dim=round(Int, cfg.head_dim * cfg.partial_rotary_factor),
    )
    layers = map(enumerate(cfg.layer_types)) do (i, kind)
        prefix = "$(P).layers.$(i - 1)"
        mixer = if kind === :full_attention
            a = "$(prefix).self_attn"
            bias = cfg.attention_bias
            GQA(
                _checkpoint_linear(weights, "$(a).q_proj"; bias),
                _checkpoint_linear(weights, "$(a).k_proj"; bias),
                _checkpoint_linear(weights, "$(a).v_proj"; bias),
                _checkpoint_linear(weights, "$(a).o_proj"; bias),
                rope;
                num_heads_q=cfg.num_attention_heads, num_heads_k=cfg.num_key_value_heads,
                head_dim=cfg.head_dim, q_norm=norm("$(a).q_norm.weight"),
                k_norm=norm("$(a).k_norm.weight"), output_gate=cfg.attn_output_gate,
            )
        elseif kind === :linear_attention
            d = "$(prefix).linear_attn"
            GatedDeltaNet(
                _checkpoint_linear(weights, "$(d).in_proj_qkv"),
                _checkpoint_linear(weights, "$(d).in_proj_z"),
                _checkpoint_linear(weights, "$(d).in_proj_b"),
                _checkpoint_linear(weights, "$(d).in_proj_a"),
                _checkpoint_linear(weights, "$(d).out_proj"),
                _dense32(weights["$(d).conv1d.weight"]),
                vector("$(d).dt_bias"), vector("$(d).A_log"), vector("$(d).norm.weight"),
                cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                cfg.linear_key_head_dim, cfg.linear_value_head_dim, eps,
            )
        else
            throw(ArgumentError("unsupported Qwen3.5 layer type `$(kind)`"))
        end
        mlp = SiLUGatedMLP(
            _checkpoint_linear(weights, "$(prefix).mlp.gate_proj"),
            _checkpoint_linear(weights, "$(prefix).mlp.up_proj"),
            _checkpoint_linear(weights, "$(prefix).mlp.down_proj"),
        )
        DecoderLayer(mixer, norm("$(prefix).input_layernorm.weight"),
            norm("$(prefix).post_attention_layernorm.weight"), mlp)
    end
    embedding = weights["$(P).embed_tokens.weight"]
    # (vocab, hidden) in HF; the file's order is the (hidden, vocab) a lookup wants.
    table = eltype(embedding) == BFloat16 ? BF16Matrix(_file_order(embedding)) :
        permutedims(_dense32(embedding), (2, 1))
    lm_head = if cfg.tie_word_embeddings
        Linear(_dense32(embedding), nothing)
    else
        _checkpoint_linear(weights, "lm_head")
    end
    model = DecoderModel(TokenEmbedding(table), layers, norm("$(P).norm.weight"))
    return Qwen3_5ForCausalLM(cfg, model, lm_head)
end

"""
    qwen3_5_state_dict_map(cfg::Qwen3_5Config) -> Dict

HuggingFace → internal parameter table for Qwen3.5's text model. Both kinds of
block sit in each layer's `self_attn` slot here; HF names the delta net
`linear_attn`. Vision and multi-token-prediction weights are not mapped.
"""
function qwen3_5_state_dict_map(cfg::Qwen3_5Config)
    P = cfg.hf_prefix
    out = Dict{String,Tuple{Tuple,Symbol}}()
    out["$(P).embed_tokens.weight"] = ((:model, :embed_tokens, :weight), :transpose)
    out["$(P).norm.weight"] = ((:model, :norm, :weight), :as_is)
    cfg.tie_word_embeddings || (out["lm_head.weight"] = ((:lm_head, :weight), :as_is))
    for (i, kind) in enumerate(cfg.layer_types)
        layer, prefix = (:model, :layers, i), "$(P).layers.$(i - 1)"
        for norm in (:input_layernorm, :post_attention_layernorm)
            out["$(prefix).$(norm).weight"] = ((layer..., norm, :weight), :as_is)
        end
        for proj in (:gate_proj, :up_proj, :down_proj)
            out["$(prefix).mlp.$(proj).weight"] = ((layer..., :mlp, proj, :weight), :as_is)
        end
        mixer = (layer..., :self_attn)
        if kind === :full_attention
            for (hf, field) in (("q_proj", :wq), ("k_proj", :wk), ("v_proj", :wv), ("o_proj", :wo))
                out["$(prefix).self_attn.$(hf).weight"] = ((mixer..., field, :weight), :as_is)
                if cfg.attention_bias
                    out["$(prefix).self_attn.$(hf).bias"] = ((mixer..., field, :bias), :as_is)
                end
            end
            out["$(prefix).self_attn.q_norm.weight"] = ((mixer..., :q_norm, :weight), :as_is)
            out["$(prefix).self_attn.k_norm.weight"] = ((mixer..., :k_norm, :weight), :as_is)
        else
            for proj in (:in_proj_qkv, :in_proj_z, :in_proj_b, :in_proj_a, :out_proj)
                out["$(prefix).linear_attn.$(proj).weight"] = ((mixer..., proj, :weight), :as_is)
            end
            out["$(prefix).linear_attn.conv1d.weight"] = ((mixer..., :conv1d), :as_is)
            out["$(prefix).linear_attn.dt_bias"] = ((mixer..., :dt_bias), :as_is)
            out["$(prefix).linear_attn.A_log"] = ((mixer..., :A_log), :as_is)
            out["$(prefix).linear_attn.norm.weight"] = ((mixer..., :norm_weight), :as_is)
        end
    end
    return out
end

"""
    load_state_dict!(lm::Qwen3_5ForCausalLM, weights) -> lm

Populate `lm` from an HF-keyed dictionary, copying the embeddings into the LM
head when the checkpoint ties them (Qwen3.5's small models do).
"""
function load_state_dict!(lm::Qwen3_5ForCausalLM, weights::AbstractDict{String,<:AbstractArray})
    load_into!(lm, weights, qwen3_5_state_dict_map(lm.config))
    if lm.config.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end
    return lm
end

"""
    build_caches(lm::Qwen3_5ForCausalLM, max_seq, batch_size; eltype=Float32)

One cache per layer: a [`KVCache`](@ref) for each attention layer and a
[`DeltaNetCache`](@ref), whose size does not depend on `max_seq`, for each
delta-net layer.
"""
function build_caches(
    lm::Qwen3_5ForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    proto = lm.model.embed_tokens.weight
    return Any[
        layer.self_attn isa GatedDeltaNet ? DeltaNetCache(layer.self_attn, batch_size) :
        KVCache(proto, cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for layer in lm.model.layers
    ]
end
