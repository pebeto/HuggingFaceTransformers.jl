"""
    MixtralConfig

Architectural hyperparameters for a Mixtral (sparse Mixture-of-Experts)
model. Field names mirror HuggingFace's `config.json` keys (Julianized
to snake_case).

Mixtral's decoder block is structurally identical to Mistral's
(`DecoderLayer` with RMSNorm + GQA + RoPE + sliding window) except the
MLP slot holds an [`MoEMLP`](@ref) instead of a single
`SiLUGatedMLP`. The MoE block routes each token to its top-K of N
experts via a softmax-renormalized gate.

`num_local_experts` and `num_experts_per_tok` map to the MoE block's
`num_experts` and `top_k` respectively. Mixtral-8x7B uses 8 experts
with top-2 routing; Mixtral-8x22B is the same shape with bigger
weights.
"""
Base.@kwdef struct MixtralConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    num_local_experts::Int
    num_experts_per_tok::Int
    rope_theta::Float64 = 1_000_000.0
    rms_norm_eps::Float64 = 1.0e-5
    tie_word_embeddings::Bool = false
    sliding_window::Union{Nothing,Int} = nothing
end

"""
    MixtralForCausalLM{C, M, H}

Top-level Mixtral container: a `DecoderModel` trunk (reused) populated
with `DecoderLayer`s whose `mlp` slot holds an [`MoEMLP`](@ref). The
LM head is a bias-less `Linear`; Mixtral checkpoints don't tie
embeddings.
"""
struct MixtralForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::MixtralForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer MixtralForCausalLM
function Flux.Optimisers.trainable(lm::MixtralForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    MixtralForCausalLM(cfg::MixtralConfig)

Materialize a fresh, randomly-initialized `MixtralForCausalLM`. Every
layer's `mlp` is an `MoEMLP` configured with `cfg.num_local_experts`
experts and `cfg.num_experts_per_tok` top-K routing.
"""
function MixtralForCausalLM(cfg::MixtralConfig)
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
            MoEMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_local_experts,
                cfg.num_experts_per_tok,
            ),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    model = DecoderModel(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        layers,
        RMSNorm(cfg.hidden_size, eps),
    )
    lm_head = Linear(cfg.hidden_size, cfg.vocab_size)
    return MixtralForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::MixtralForCausalLM, max_seq, batch_size; eltype=Float32)
"""
function build_caches(
    lm::MixtralForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    mixtral_state_dict_map(cfg::MixtralConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal parameter table. Attention and norm entries
match the Llama/Mistral convention; the MoE block contributes one
`gate.weight` per layer plus three weights per expert (HF's `w1` /
`w2` / `w3` map to our `gate_proj` / `down_proj` / `up_proj`).
"""
function mixtral_state_dict_map(cfg::MixtralConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["model.embed_tokens.weight"] = ((:model, :embed_tokens, :weight), :transpose)

    for i in 0:(cfg.num_hidden_layers - 1)
        layer_path = (:model, :layers, i + 1)
        prefix = "model.layers.$(i)"

        out["$(prefix).input_layernorm.weight"] = (
            (layer_path..., :input_layernorm, :weight), :as_is
        )
        out["$(prefix).self_attn.q_proj.weight"] = (
            (layer_path..., :self_attn, :wq, :weight), :as_is
        )
        out["$(prefix).self_attn.k_proj.weight"] = (
            (layer_path..., :self_attn, :wk, :weight), :as_is
        )
        out["$(prefix).self_attn.v_proj.weight"] = (
            (layer_path..., :self_attn, :wv, :weight), :as_is
        )
        out["$(prefix).self_attn.o_proj.weight"] = (
            (layer_path..., :self_attn, :wo, :weight), :as_is
        )
        out["$(prefix).post_attention_layernorm.weight"] = (
            (layer_path..., :post_attention_layernorm, :weight), :as_is
        )

        # MoE: router gate + per-expert SwiGLU weights.
        out["$(prefix).block_sparse_moe.gate.weight"] = (
            (layer_path..., :mlp, :gate, :weight), :as_is
        )
        for e in 0:(cfg.num_local_experts - 1)
            ep = (layer_path..., :mlp, :experts, e + 1)
            # HF Mixtral's w1/w2/w3 = gate_proj/down_proj/up_proj. Yes, the
            # ordering is non-obvious — w2 is the down-projection.
            out["$(prefix).block_sparse_moe.experts.$(e).w1.weight"] = (
                (ep..., :gate_proj, :weight), :as_is
            )
            out["$(prefix).block_sparse_moe.experts.$(e).w2.weight"] = (
                (ep..., :down_proj, :weight), :as_is
            )
            out["$(prefix).block_sparse_moe.experts.$(e).w3.weight"] = (
                (ep..., :up_proj, :weight), :as_is
            )
        end
    end

    out["model.norm.weight"] = ((:model, :norm, :weight), :as_is)

    if !cfg.tie_word_embeddings
        out["lm_head.weight"] = ((:lm_head, :weight), :as_is)
    end

    return out
end

"""
    load_state_dict!(lm::MixtralForCausalLM, weights) -> lm

Populate `lm` in-place. Pure delegation to the generic loader plus
tied-embedding materialization if the config requests it (no Mixtral
checkpoint ties, but the path is supported for symmetry with other
models).
"""
function load_state_dict!(
    lm::MixtralForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(lm, weights, mixtral_state_dict_map(lm.config))

    if lm.config.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
