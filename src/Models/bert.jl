"""
    BertConfig

Architectural hyperparameters for a BERT- or RoBERTa-family encoder.
Field names mirror HuggingFace's `config.json` keys (Julianized to
snake_case).

BERT and RoBERTa are encoder-only transformers; they differ from
decoder-only models in three architecturally significant ways:

- **Bidirectional attention**: no causal mask. Every query attends to
  every key. Wired up via the `causal=false` flag on `GQA`.
- **Post-norm** decoder block: LayerNorm comes *after* the residual add,
  not before. See [`BertEncoderLayer`](@ref).
- **Three summed embeddings**: token, position, and token-type. The
  sum is then normalized by a LayerNorm before entering the encoder
  stack.

`hf_prefix` controls state-dict naming: `"bert"` (default) for BERT
checkpoints, `"roberta"` for RoBERTa. `head_prefix` similarly selects
the prediction-head namespace: `"cls.predictions"` for BERT,
`"lm_head"` for RoBERTa. `position_embedding_offset` is `0` for BERT
and `2` for RoBERTa (RoBERTa offsets positions by `pad_token_id + 1`).
"""
Base.@kwdef struct BertConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    max_position_embeddings::Int
    type_vocab_size::Int = 2
    layer_norm_eps::Float64 = 1.0e-12
    pad_token_id::Int = 0
    position_embedding_offset::Int = 0
    hf_prefix::String = "bert"
    head_prefix::String = "cls.predictions"
    tie_word_embeddings::Bool = true
end

"""
    BertEmbeddings{TE, PE, TYE, N}

Token + position + token-type embedding sum, followed by a LayerNorm.
The HF-equivalent state-dict path is
`{hf_prefix}.embeddings.{word,position,token_type}_embeddings.weight`
and `{hf_prefix}.embeddings.LayerNorm.{weight,bias}`.
"""
struct BertEmbeddings{TE,PE,TYE,N}
    embed_tokens::TE
    embed_positions::PE
    embed_types::TYE
    norm::N
    position_offset::Int
end

function (e::BertEmbeddings)(
    input_ids::AbstractMatrix{<:Integer},
    position_ids::AbstractVector{<:Integer},
    token_type_ids::AbstractMatrix{<:Integer},
)
    tok = e.embed_tokens(input_ids)          # (hidden, seq, batch)
    # RoBERTa reserves the first `pad_token_id + 1` rows of the position table, so
    # its positions start at 2 rather than 0. The table is allocated with those
    # extra rows, and this is where the shift is applied.
    pos = e.embed_positions(position_ids .+ e.position_offset)
    typ = e.embed_types(token_type_ids)      # (hidden, seq, batch)
    h = tok .+ reshape(pos, size(pos, 1), size(pos, 2), 1) .+ typ
    return e.norm(h)
end

Flux.@layer BertEmbeddings

"""
    BertEncoderLayer{A, AN, IM, OM, ON}

Post-norm encoder block:

```
attn = self_attn(x)
x = attn_norm(x + attn)
ffn = output_dense(gelu(intermediate(x)))
x = ffn_norm(x + ffn)
```

This is the inverse of Llama's pre-norm: LayerNorm sits *after* each
residual add, not before. `self_attn` is bidirectional (no causal
mask). The FFN is non-gated GELU (exact, via `GeluMLP` with
`approx=false`).
"""
struct BertEncoderLayer{A,AN,IM,OM,ON}
    self_attn::A
    attn_norm::AN
    intermediate::IM
    output_dense::OM
    ffn_norm::ON
end

function (layer::BertEncoderLayer)(x::AbstractArray; position_ids=nothing)
    # Self-attention with residual + post-norm.
    a = layer.self_attn(x; position_ids=position_ids)
    x = layer.attn_norm(x .+ a)

    # FFN with residual + post-norm. The MLP is split here (intermediate
    # / output_dense) rather than wrapped in `GeluMLP` so the state-dict
    # paths land directly on the per-HF-key Julia paths.
    h = layer.intermediate(x)
    h = _gelu_exact.(h)
    h = layer.output_dense(h)
    return layer.ffn_norm(x .+ h)
end

Flux.@layer BertEncoderLayer

"""
    BertEncoder{L}

Stack of `BertEncoderLayer`s. Sequence position information is
already in the input via [`BertEmbeddings`](@ref), so the trunk just
threads `position_ids` through for any layer that wants it.
"""
struct BertEncoder{L}
    layers::L
end

function (e::BertEncoder)(x::AbstractArray; position_ids=nothing)
    h = x
    for layer in e.layers
        h = layer(h; position_ids=position_ids)
    end
    return h
end

Flux.@layer BertEncoder

"""
    BertModel{E, EN}

Encoder-only transformer trunk: embeddings → encoder stack. Returns
the per-token hidden states (no pooler, since sequence classification heads
are out of scope for now).
"""
struct BertModel{E,EN}
    embeddings::E
    encoder::EN
end

function (m::BertModel)(
    input_ids::AbstractMatrix{<:Integer};
    token_type_ids::Union{Nothing,AbstractMatrix{<:Integer}}=nothing,
    position_ids::Union{Nothing,AbstractVector{<:Integer}}=nothing,
)
    seq_len = size(input_ids, 1)
    if isnothing(token_type_ids)
        token_type_ids = zeros(Int, size(input_ids))
    end
    if isnothing(position_ids)
        position_ids = collect(0:(seq_len - 1))
    end
    h = m.embeddings(input_ids, position_ids, token_type_ids)
    return m.encoder(h; position_ids=position_ids)
end

Flux.@layer BertModel

"""
    BertModel(cfg::BertConfig)

Materialize a fresh, randomly-initialized encoder trunk (embeddings +
encoder stack) matching `cfg`. Shared by [`BertForMaskedLM`](@ref) and
[`BertEmbeddingModel`](@ref).
"""
function BertModel(cfg::BertConfig)
    head_dim = cfg.hidden_size ÷ cfg.num_attention_heads
    eps = Float32(cfg.layer_norm_eps)

    embeddings = BertEmbeddings(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        TokenEmbedding(
            cfg.max_position_embeddings + cfg.position_embedding_offset, cfg.hidden_size
        ),
        TokenEmbedding(cfg.type_vocab_size, cfg.hidden_size),
        LayerNorm(cfg.hidden_size, eps),
        cfg.position_embedding_offset,
    )

    layers = [
        BertEncoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_attention_heads,    # MHA
                head_dim,
                nothing;                    # no RoPE — positions live in the embedding
                qkv_bias=true,
                wo_bias=true,
                causal=false,               # bidirectional
            ),
            LayerNorm(cfg.hidden_size, eps),
            Linear(cfg.hidden_size, cfg.intermediate_size; bias=true),
            Linear(cfg.intermediate_size, cfg.hidden_size; bias=true),
            LayerNorm(cfg.hidden_size, eps),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    return BertModel(embeddings, BertEncoder(layers))
end

"""
    BertLMHead{D, N, P, B}

MaskedLM prediction head: `dense → gelu → LayerNorm → decoder + bias`.
`decoder.weight` is tied to the input embedding at load time. `bias` is a type
parameter so the vector can hold a device array after a GPU move.
"""
mutable struct BertLMHead{D,N,P,B}
    dense::D
    norm::N
    decoder::P
    bias::B
end

function (h::BertLMHead)(x::AbstractArray)
    y = h.dense(x)
    y = _gelu_exact.(y)
    y = h.norm(y)
    y = h.decoder(y)
    return y .+ h.bias
end

Flux.@layer BertLMHead

"""
    BertForMaskedLM{C, M, H}

Top-level BERT/RoBERTa MaskedLM container: `BertModel` trunk plus a
`BertLMHead` that projects hidden states back to vocabulary logits.
"""
struct BertForMaskedLM{C,M,H}
    config::C
    trunk::M
    head::H
end

function (lm::BertForMaskedLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.trunk(input_ids; kwargs...)
    return lm.head(h)
end

Flux.@layer BertForMaskedLM
function Flux.Optimisers.trainable(lm::BertForMaskedLM)
    return (; trunk=lm.trunk, head=lm.head)
end

"""
    BertForMaskedLM(cfg::BertConfig)

Materialize a fresh, randomly-initialized `BertForMaskedLM` matching the
shapes in `cfg`. Every Linear in the encoder block carries bias, the
embedding LayerNorm and post-norm LayerNorms all carry bias, and the
MLP uses exact (erf) GELU.
"""
function BertForMaskedLM(cfg::BertConfig)
    eps = Float32(cfg.layer_norm_eps)
    trunk = BertModel(cfg)

    head = BertLMHead(
        Linear(cfg.hidden_size, cfg.hidden_size; bias=true),
        LayerNorm(cfg.hidden_size, eps),
        Linear(cfg.hidden_size, cfg.vocab_size),  # tied to embed_tokens at load time
        zeros(Float32, cfg.vocab_size),
    )

    return BertForMaskedLM(cfg, trunk, head)
end

"""
    bert_state_dict_map(cfg::BertConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal parameter table. Honors `cfg.hf_prefix` (which
namespace the trunk lives under: `bert.…` or `roberta.…`) and
`cfg.head_prefix` (the prediction head's namespace:
`cls.predictions.…` for BERT, `lm_head.…` for RoBERTa).

`{prefix}.embeddings.*.weight` entries transpose on load (HF stores
embeddings as `(vocab, hidden)`; we want `(hidden, vocab)`). All other
linear weights are stored in standard `(out, in)` orientation, so no
transpose is needed.
"""
function bert_state_dict_map(cfg::BertConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()
    p = cfg.hf_prefix
    hp = cfg.head_prefix
    # Sentence-embedding checkpoints (BGE, E5) often store the trunk with no
    # `bert.`/`roberta.` namespace; `pfx` drops the separator when empty.
    pfx(s) = isempty(p) ? s : "$(p).$(s)"

    # Embeddings: word / position / token-type embeddings each (vocab-sized, hidden).
    out[pfx("embeddings.word_embeddings.weight")] = (
        (:trunk, :embeddings, :embed_tokens, :weight), :transpose
    )
    out[pfx("embeddings.position_embeddings.weight")] = (
        (:trunk, :embeddings, :embed_positions, :weight), :transpose
    )
    out[pfx("embeddings.token_type_embeddings.weight")] = (
        (:trunk, :embeddings, :embed_types, :weight), :transpose
    )
    out[pfx("embeddings.LayerNorm.weight")] = (
        (:trunk, :embeddings, :norm, :weight), :as_is
    )
    out[pfx("embeddings.LayerNorm.bias")] = ((:trunk, :embeddings, :norm, :bias), :as_is)

    for i in 0:(cfg.num_hidden_layers - 1)
        layer_path = (:trunk, :encoder, :layers, i + 1)
        prefix = pfx("encoder.layer.$(i)")

        # Self-attention Q/K/V (each shipped as its own linear; not fused for BERT).
        out["$(prefix).attention.self.query.weight"] = (
            (layer_path..., :self_attn, :wq, :weight), :as_is
        )
        out["$(prefix).attention.self.query.bias"] = (
            (layer_path..., :self_attn, :wq, :bias), :as_is
        )
        out["$(prefix).attention.self.key.weight"] = (
            (layer_path..., :self_attn, :wk, :weight), :as_is
        )
        out["$(prefix).attention.self.key.bias"] = (
            (layer_path..., :self_attn, :wk, :bias), :as_is
        )
        out["$(prefix).attention.self.value.weight"] = (
            (layer_path..., :self_attn, :wv, :weight), :as_is
        )
        out["$(prefix).attention.self.value.bias"] = (
            (layer_path..., :self_attn, :wv, :bias), :as_is
        )

        # Output projection + post-attention LayerNorm.
        out["$(prefix).attention.output.dense.weight"] = (
            (layer_path..., :self_attn, :wo, :weight), :as_is
        )
        out["$(prefix).attention.output.dense.bias"] = (
            (layer_path..., :self_attn, :wo, :bias), :as_is
        )
        out["$(prefix).attention.output.LayerNorm.weight"] = (
            (layer_path..., :attn_norm, :weight), :as_is
        )
        out["$(prefix).attention.output.LayerNorm.bias"] = (
            (layer_path..., :attn_norm, :bias), :as_is
        )

        # FFN (intermediate + output) + post-FFN LayerNorm.
        out["$(prefix).intermediate.dense.weight"] = (
            (layer_path..., :intermediate, :weight), :as_is
        )
        out["$(prefix).intermediate.dense.bias"] = (
            (layer_path..., :intermediate, :bias), :as_is
        )
        out["$(prefix).output.dense.weight"] = (
            (layer_path..., :output_dense, :weight), :as_is
        )
        out["$(prefix).output.dense.bias"] = ((layer_path..., :output_dense, :bias), :as_is)
        out["$(prefix).output.LayerNorm.weight"] = (
            (layer_path..., :ffn_norm, :weight), :as_is
        )
        out["$(prefix).output.LayerNorm.bias"] = ((layer_path..., :ffn_norm, :bias), :as_is)
    end

    # Prediction head. The BERT and RoBERTa keys differ in two places:
    # the head namespace prefix, and the inner LayerNorm's filename
    # (`transform.LayerNorm` vs `layer_norm`). Both write `decoder.weight`
    # and `bias` at the head root.
    if cfg.head_prefix == "cls.predictions"
        out["$(hp).transform.dense.weight"] = ((:head, :dense, :weight), :as_is)
        out["$(hp).transform.dense.bias"] = ((:head, :dense, :bias), :as_is)
        out["$(hp).transform.LayerNorm.weight"] = ((:head, :norm, :weight), :as_is)
        out["$(hp).transform.LayerNorm.bias"] = ((:head, :norm, :bias), :as_is)
    else
        # RoBERTa: flat layout under `lm_head`.
        out["$(hp).dense.weight"] = ((:head, :dense, :weight), :as_is)
        out["$(hp).dense.bias"] = ((:head, :dense, :bias), :as_is)
        out["$(hp).layer_norm.weight"] = ((:head, :norm, :weight), :as_is)
        out["$(hp).layer_norm.bias"] = ((:head, :norm, :bias), :as_is)
    end
    out["$(hp).bias"] = ((:head, :bias), :as_is)

    # decoder.weight is materialized from the input embedding at load time
    # when `tie_word_embeddings` is true (the default for both checkpoints).
    if !cfg.tie_word_embeddings
        out["$(hp).decoder.weight"] = ((:head, :decoder, :weight), :as_is)
    end

    return out
end

# Checkpoints converted before transformers renamed LayerNorm's parameters still
# ship `LayerNorm.gamma` / `LayerNorm.beta`; `bert-base-uncased` is one of them.
# HF renames these on load and so do we, scoped to `LayerNorm.` so that a model
# using `gamma` for something else (LayerScale, say) is left alone.
const _LEGACY_NORM_NAMES = ("LayerNorm.gamma" => "LayerNorm.weight",
    "LayerNorm.beta" => "LayerNorm.bias")

function _bert_rename_legacy_norms(weights::AbstractDict{String,<:AbstractArray})
    needs_rename = any(keys(weights)) do key
        any(pair -> endswith(key, first(pair)), _LEGACY_NORM_NAMES)
    end
    needs_rename || return weights

    renamed = Dict{String,AbstractArray}()
    for (key, value) in weights
        new_key = key
        for (old, new) in _LEGACY_NORM_NAMES
            if endswith(key, old)
                new_key = key[1:(end - length(old))] * new
                break
            end
        end
        renamed[new_key] = value
    end
    return renamed
end

"""
    load_state_dict!(lm::BertForMaskedLM, weights) -> lm

Populate `lm` in-place. The encoder Q/K/V projections are not fused in
BERT/RoBERTa (unlike GPT-2/NeoX/Phi-3), so the generic table handles
the whole load except the tied LM head: `decoder.weight` is the
input-embedding transpose.
"""
function load_state_dict!(
    lm::BertForMaskedLM, weights::AbstractDict{String,<:AbstractArray}
)
    weights = _bert_rename_legacy_norms(weights)
    load_into!(lm, weights, bert_state_dict_map(lm.config))

    if lm.config.tie_word_embeddings
        copyto!(
            lm.head.decoder.weight,
            permutedims(lm.trunk.embeddings.embed_tokens.weight, (2, 1)),
        )
    end

    return lm
end
