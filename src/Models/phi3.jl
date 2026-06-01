"""
    Phi3Config

Architectural hyperparameters for a Phi-3 model. Field names mirror
HuggingFace's `config.json` keys (Julianized to snake_case).

Phi-3 is structurally close to Llama (RMSNorm, SwiGLU MLP, GQA, plain
RoPE) but ships its weights in two fused tensors per layer:
`self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight`. The model
struct here reuses `DecoderModel` / `DecoderLayer` / `GQA` /
`SiLUGatedMLP` unchanged; the slicing happens once in
[`load_state_dict!`](@ref load_state_dict!(::Phi3ForCausalLM, ::AbstractDict)).
The future Phase 4 path is to swap in a single fused matmul instead;
correctness-first today.

Only `partial_rotary_factor == 1.0` (the Phi-3 / Phi-3.5 baseline) is
supported. The loader rejects other values explicitly — `longrope`
scaling and partial RoPE need their own implementation, deferred.
"""
Base.@kwdef struct Phi3Config
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
    Phi3ForCausalLM{C, M, H}

`DecoderModel` trunk wrapped in a `Linear` LM head — the same shape as
`LlamaForCausalLM`, distinguished by config type so dispatch picks the
right loader.
"""
struct Phi3ForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::Phi3ForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer Phi3ForCausalLM
function Flux.Optimisers.trainable(lm::Phi3ForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    Phi3ForCausalLM(cfg::Phi3Config)

Materialize a fresh, randomly-initialized `Phi3ForCausalLM` matching the
shapes in `cfg`. Sliding-window attention is threaded through to every
`GQA` layer when `cfg.sliding_window` is set.
"""
function Phi3ForCausalLM(cfg::Phi3Config)
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
    return Phi3ForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::Phi3ForCausalLM, max_seq, batch_size; eltype=Float32)
"""
function build_caches(
    lm::Phi3ForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    phi3_state_dict_map(cfg::Phi3Config) -> Dict{String, Tuple{Tuple, Symbol}}

Pure-data state-dict map for the entries Phi-3 stores 1:1 with the
internal model: layernorms, `o_proj`, `down_proj`, embeddings,
final norm, and (when untied) `lm_head`. The fused `qkv_proj` and
`gate_up_proj` weights are NOT in this table — they're handled by the
slicing pass in [`load_state_dict!`](@ref).
"""
function phi3_state_dict_map(cfg::Phi3Config)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["model.embed_tokens.weight"] = ((:model, :embed_tokens, :weight), :transpose)

    for i in 0:(cfg.num_hidden_layers - 1)
        layer_path = (:model, :layers, i + 1)
        hf_prefix = "model.layers.$(i)"

        out["$(hf_prefix).input_layernorm.weight"] = (
            (layer_path..., :input_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).self_attn.o_proj.weight"] = (
            (layer_path..., :self_attn, :wo, :weight), :as_is
        )
        out["$(hf_prefix).post_attention_layernorm.weight"] = (
            (layer_path..., :post_attention_layernorm, :weight), :as_is
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
    load_state_dict!(lm::Phi3ForCausalLM, weights) -> lm

Populate `lm` in-place. Runs the standard pure-data loader for unfused
weights, then unpacks each layer's `qkv_proj` and `gate_up_proj` into
`(wq, wk, wv)` and `(gate_proj, up_proj)` respectively. HF's convention:
QKV is concatenated along the output dimension as `[Q; K; V]`; gate-up
is `[gate; up]`.
"""
function load_state_dict!(
    lm::Phi3ForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    cfg = lm.config
    load_into!(lm, weights, phi3_state_dict_map(cfg))

    n_q = cfg.num_attention_heads * cfg.head_dim
    n_kv = cfg.num_key_value_heads * cfg.head_dim
    inter = cfg.intermediate_size

    for i in 0:(cfg.num_hidden_layers - 1)
        layer = lm.model.layers[i + 1]
        attn = layer.self_attn

        qkv_key = "model.layers.$(i).self_attn.qkv_proj.weight"
        haskey(weights, qkv_key) || throw(KeyError("missing fused QKV weight: $(qkv_key)"))
        qkv = weights[qkv_key]
        size(qkv, 1) == n_q + 2 * n_kv || throw(
            DimensionMismatch(
                "qkv_proj.weight first dim is $(size(qkv, 1)), expected " *
                "$(n_q + 2 * n_kv) (= num_q*head_dim + 2*num_kv*head_dim)",
            ),
        )
        copyto!(attn.wq.weight, @view qkv[1:n_q, :])
        copyto!(attn.wk.weight, @view qkv[(n_q + 1):(n_q + n_kv), :])
        copyto!(attn.wv.weight, @view qkv[(n_q + n_kv + 1):(n_q + 2 * n_kv), :])

        gup_key = "model.layers.$(i).mlp.gate_up_proj.weight"
        haskey(weights, gup_key) ||
            throw(KeyError("missing fused gate-up weight: $(gup_key)"))
        gup = weights[gup_key]
        size(gup, 1) == 2 * inter || throw(
            DimensionMismatch(
                "gate_up_proj.weight first dim is $(size(gup, 1)), expected " *
                "$(2 * inter) (= 2 * intermediate_size)",
            ),
        )
        copyto!(layer.mlp.gate_proj.weight, @view gup[1:inter, :])
        copyto!(layer.mlp.up_proj.weight, @view gup[(inter + 1):(2 * inter), :])
    end

    if cfg.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
