"""
    NomicBertConfig

Architecture for nomic-embed-text (NomicBERT). A BERT-postnorm encoder with
three swaps from vanilla BERT: rotary embeddings instead of an absolute
position table, a fused QKV projection with no biases, and a SwiGLU MLP. Field
names follow the relevant `config.json` keys. `rope_theta` is nomic's
`rotary_emb_base` (1000, not Llama's 10000).
"""
Base.@kwdef struct NomicBertConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    head_dim::Int
    type_vocab_size::Int = 2
    layer_norm_eps::Float64 = 1.0e-12
    rope_theta::Float64 = 1000.0
    pad_token_id::Int = 0
end

"""
    NomicBertEmbeddings{TE, TYE, N}

Word + token-type embedding sum followed by a LayerNorm (`emb_ln`). NomicBERT
carries no position embeddings; rotary handles position inside attention.
"""
struct NomicBertEmbeddings{TE,TYE,N}
    embed_tokens::TE
    embed_types::TYE
    norm::N
end

function (e::NomicBertEmbeddings)(
    input_ids::AbstractMatrix{<:Integer}, token_type_ids::AbstractMatrix{<:Integer}
)
    h = e.embed_tokens(input_ids) .+ e.embed_types(token_type_ids)
    return e.norm(h)
end

Flux.@layer NomicBertEmbeddings

"""
    NomicBertLayer{A, N1, M, N2}

Post-norm encoder block: `x = norm1(x + attn(x)); x = norm2(x + mlp(x))`. Same
residual order as `BertEncoderLayer`, but `self_attn` carries rotary embeddings
and the MLP is gated SwiGLU.
"""
struct NomicBertLayer{A,N1,M,N2}
    self_attn::A
    norm1::N1
    mlp::M
    norm2::N2
end

function (layer::NomicBertLayer)(x::AbstractArray; position_ids=nothing)
    a = layer.self_attn(x; position_ids=position_ids)
    x = layer.norm1(x .+ a)
    h = layer.mlp(x)
    return layer.norm2(x .+ h)
end

Flux.@layer NomicBertLayer

"""
    NomicBertModel{E, L}

Encoder trunk: embeddings → stack of `NomicBertLayer`s. Returns per-token
hidden states. Post-norm means the last block's `norm2` is the final
normalization, so there is no separate trailing LayerNorm.
"""
struct NomicBertModel{E,L}
    embeddings::E
    layers::L
end

function (m::NomicBertModel)(
    input_ids::AbstractMatrix{<:Integer};
    token_type_ids::Union{Nothing,AbstractMatrix{<:Integer}}=nothing,
    position_ids::Union{Nothing,AbstractVector{<:Integer}}=nothing,
)
    seq_len = size(input_ids, 1)
    if isnothing(token_type_ids)
        token_type_ids = zeros(Int, size(input_ids))
    end
    pos = isnothing(position_ids) ? collect(0:(seq_len - 1)) : position_ids
    h = m.embeddings(input_ids, token_type_ids)
    for layer in m.layers
        h = layer(h; position_ids=pos)
    end
    return h
end

Flux.@layer NomicBertModel

"""
    NomicBertModel(cfg::NomicBertConfig)

Materialize a fresh NomicBERT trunk. Attention is bidirectional MHA with full
rotary at `cfg.rope_theta`; the MLP is `SiLUGatedMLP`; every projection is
bias-free.
"""
function NomicBertModel(cfg::NomicBertConfig)
    eps = Float32(cfg.layer_norm_eps)
    embeddings = NomicBertEmbeddings(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        TokenEmbedding(cfg.type_vocab_size, cfg.hidden_size),
        LayerNorm(cfg.hidden_size, eps),
    )
    layers = [
        NomicBertLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_attention_heads,        # MHA
                cfg.head_dim,
                RoPE(cfg.head_dim; base=cfg.rope_theta);
                qkv_bias=false,
                wo_bias=false,
                causal=false,                   # bidirectional
            ),
            LayerNorm(cfg.hidden_size, eps),
            SiLUGatedMLP(cfg.hidden_size, cfg.intermediate_size),
            LayerNorm(cfg.hidden_size, eps),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]
    return NomicBertModel(embeddings, layers)
end

"""
    nomic_state_dict_map(cfg::NomicBertConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal parameter table for everything except the fused QKV,
which `load_state_dict!` splits by hand. The gated-MLP keys map `fc12` → gate,
`fc11` → up, `fc2` → down (the product `fc11 * silu(fc12)` matches
`SiLUGatedMLP`'s `up * silu(gate)`). Embedding tables transpose on load.
"""
function nomic_state_dict_map(cfg::NomicBertConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()
    out["embeddings.word_embeddings.weight"] = (
        (:trunk, :embeddings, :embed_tokens, :weight), :transpose
    )
    out["embeddings.token_type_embeddings.weight"] = (
        (:trunk, :embeddings, :embed_types, :weight), :transpose
    )
    out["emb_ln.weight"] = ((:trunk, :embeddings, :norm, :weight), :as_is)
    out["emb_ln.bias"] = ((:trunk, :embeddings, :norm, :bias), :as_is)

    for i in 0:(cfg.num_hidden_layers - 1)
        lp = (:trunk, :layers, i + 1)
        p = "encoder.layers.$(i)"
        out["$(p).norm1.weight"] = ((lp..., :norm1, :weight), :as_is)
        out["$(p).norm1.bias"] = ((lp..., :norm1, :bias), :as_is)
        out["$(p).attn.out_proj.weight"] = ((lp..., :self_attn, :wo, :weight), :as_is)
        out["$(p).mlp.fc11.weight"] = ((lp..., :mlp, :up_proj, :weight), :as_is)
        out["$(p).mlp.fc12.weight"] = ((lp..., :mlp, :gate_proj, :weight), :as_is)
        out["$(p).mlp.fc2.weight"] = ((lp..., :mlp, :down_proj, :weight), :as_is)
        out["$(p).norm2.weight"] = ((lp..., :norm2, :weight), :as_is)
        out["$(p).norm2.bias"] = ((lp..., :norm2, :bias), :as_is)
    end
    return out
end

"""
    BertEmbeddingModel(cfg::NomicBertConfig; pooling = :mean, normalize = true)

Materialize a nomic-embed model. nomic uses mean pooling, so `pooling` defaults
to `:mean`.
"""
function BertEmbeddingModel(
    cfg::NomicBertConfig; pooling::Symbol=:mean, normalize::Bool=true
)
    pooling in (:cls, :mean) ||
        throw(ArgumentError("pooling must be :cls or :mean, got :$(pooling)"))
    return BertEmbeddingModel(cfg, NomicBertModel(cfg), pooling, normalize)
end

"""
    load_state_dict!(m::BertEmbeddingModel{C, <:NomicBertModel}, weights) -> m

Load a NomicBERT trunk. The non-fused parameters go through
[`nomic_state_dict_map`](@ref); the fused `Wqkv` (rows `[Q; K; V]`) is split by
hand into the GQA's `wq`/`wk`/`wv`.
"""
function load_state_dict!(
    m::BertEmbeddingModel{C,<:NomicBertModel}, weights::AbstractDict{String,<:AbstractArray}
) where {C}
    load_into!(m, weights, nomic_state_dict_map(m.config))

    h = m.config.hidden_size
    for i in 0:(m.config.num_hidden_layers - 1)
        Wqkv = weights["encoder.layers.$(i).attn.Wqkv.weight"]   # (3h, h)
        attn = m.trunk.layers[i + 1].self_attn
        copyto!(attn.wq.weight, Wqkv[1:h, :])
        copyto!(attn.wk.weight, Wqkv[(h + 1):(2h), :])
        copyto!(attn.wv.weight, Wqkv[(2h + 1):(3h), :])
    end
    return m
end
