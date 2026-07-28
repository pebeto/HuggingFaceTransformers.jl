# Generic decoder primitives shared across the Llama-family models
# (Llama, Mistral, Qwen, Phi-3). Gemma has its own decoder layer + model
# because its block structure differs (four RMSNorms per layer, embedding
# scaling); see `gemma.jl`.

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
    DecoderLayer{A, N1, N2, M}

A standard pre-norm transformer block: norm → attention → residual,
then norm → MLP → residual. Two RMSNorms per layer.

Used unchanged by Llama, Mistral, Qwen, and Phi-3. Gemma2 needs four
norms per block and has its own `GemmaDecoderLayer`.
"""
struct DecoderLayer{A,N1,N2,M}
    self_attn::A
    input_layernorm::N1
    post_attention_layernorm::N2
    mlp::M
end

function (layer::DecoderLayer)(
    x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing
)
    h = layer.input_layernorm(x)
    h = layer.self_attn(h; cache=cache, step=step, position_ids=position_ids)
    x = x .+ h
    h = layer.post_attention_layernorm(x)
    h = layer.mlp(h)
    return x .+ h
end

Flux.@layer DecoderLayer

"""
    DecoderModel{E, L, N}

Decoder-only transformer trunk: token embedding → stack of
`DecoderLayer`s → final `RMSNorm`. Returns hidden states without a
projection to vocabulary, because the LM head lives in the per-model
`*ForCausalLM` wrappers.
"""
struct DecoderModel{E,L,N}
    embed_tokens::E
    layers::L
    norm::N
end

function (m::DecoderModel)(
    input_ids::AbstractMatrix{<:Integer}; caches=nothing, step=nothing, position_ids=nothing
)
    h = m.embed_tokens(input_ids)
    return forward_embeds(m, h; caches=caches, step=step, position_ids=position_ids)
end

"""
    forward_embeds(m::DecoderModel, inputs_embeds; caches, step, position_ids) -> hidden

Run the decoder layers and final norm on precomputed `(hidden, seq, batch)`
embeddings, skipping token lookup. Used by multimodal models (LLaVA) that splice
image features into the embedding stream.
"""
function forward_embeds(
    m::DecoderModel, h::AbstractArray; caches=nothing, step=nothing, position_ids=nothing
)
    for i in eachindex(m.layers)
        cache_i = isnothing(caches) ? nothing : caches[i]
        h = m.layers[i](h; cache=cache_i, step=step, position_ids=position_ids)
    end
    return m.norm(h)
end

Flux.@layer DecoderModel

"""
    _decoder_state_dict_map(num_hidden_layers, tie_word_embeddings; qkv_bias=false) -> Dict

Pure-data state-dict map for any HF decoder-only LM whose parameter naming
matches the Llama / Mistral / Qwen convention (`model.embed_tokens.weight`,
`model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
`model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`,
`model.norm.weight`, optional `lm_head.weight`).

`qkv_bias=true` additionally emits entries for `q_proj.bias`, `k_proj.bias`,
and `v_proj.bias`, which Qwen2 and Qwen2.5 need because they bias QKV but not
the output projection.

[`llama_state_dict_map`](@ref), [`mistral_state_dict_map`](@ref), and
[`qwen_state_dict_map`](@ref) delegate here so a fix to the path table benefits
every consumer. Phi-3 also delegates here for its non-fused entries, handling
the fused `qkv_proj` and `gate_up_proj` separately at load time.
"""
function _decoder_state_dict_map(
    num_hidden_layers::Integer, tie_word_embeddings::Bool; qkv_bias::Bool=false
)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["model.embed_tokens.weight"] = ((:model, :embed_tokens, :weight), :transpose)

    for i in 0:(num_hidden_layers - 1)
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
        if qkv_bias
            out["$(hf_prefix).self_attn.q_proj.bias"] = (
                (layer_path..., :self_attn, :wq, :bias), :as_is
            )
            out["$(hf_prefix).self_attn.k_proj.bias"] = (
                (layer_path..., :self_attn, :wk, :bias), :as_is
            )
            out["$(hf_prefix).self_attn.v_proj.bias"] = (
                (layer_path..., :self_attn, :wv, :bias), :as_is
            )
        end
        out["$(hf_prefix).post_attention_layernorm.weight"] = (
            (layer_path..., :post_attention_layernorm, :weight), :as_is
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

    if !tie_word_embeddings
        out["lm_head.weight"] = ((:lm_head, :weight), :as_is)
    end

    return out
end
