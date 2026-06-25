"""
    mean_pool(hidden) -> AbstractMatrix

Mean over the sequence dimension of `(hidden, seq, batch)` token states,
returning `(hidden, batch)`. Encoding one text per call leaves no padding, so
this equals sentence-transformers' attention-masked mean.
"""
mean_pool(h::AbstractArray) = dropdims(sum(h; dims=2); dims=2) ./ size(h, 2)

"""
    cls_pool(hidden) -> AbstractMatrix

Take the first ([CLS]) token of `(hidden, seq, batch)` states → `(hidden, batch)`.
"""
cls_pool(h::AbstractArray) = h[:, 1, :]

"""
    l2_normalize(x; dims=1) -> AbstractArray

Scale along `dims` to unit L2 norm, turning cosine similarity into a dot
product. A tiny epsilon guards the all-zero column.
"""
l2_normalize(x::AbstractArray; dims=1) = x ./ sqrt.(sum(abs2, x; dims=dims) .+ eps(Float32))

"""
    BertEmbeddingModel{C, M}

Text-embedding model: a `BertModel` trunk plus a pooling strategy and optional
L2 normalization. Covers the BGE / E5 family (vanilla BERT/RoBERTa encoders).
`pooling` is `:cls` (BGE) or `:mean` (E5); `normalize` defaults to `true`, since
retrieval checkpoints expect unit-norm vectors.
"""
struct BertEmbeddingModel{C,M}
    config::C
    trunk::M
    pooling::Symbol
    normalize::Bool
end

"""
    BertEmbeddingModel(cfg::BertConfig; pooling = :cls, normalize = true)

Materialize a fresh embedding model. `pooling` must be `:cls` or `:mean`.
"""
function BertEmbeddingModel(cfg::BertConfig; pooling::Symbol=:cls, normalize::Bool=true)
    pooling in (:cls, :mean) ||
        throw(ArgumentError("pooling must be :cls or :mean, got :$(pooling)"))
    return BertEmbeddingModel(cfg, BertModel(cfg), pooling, normalize)
end

function (m::BertEmbeddingModel)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = m.trunk(input_ids; kwargs...)            # (hidden, seq, batch)
    pooled = m.pooling === :cls ? cls_pool(h) : mean_pool(h)
    return m.normalize ? l2_normalize(pooled) : pooled
end

Flux.@layer BertEmbeddingModel
Flux.Optimisers.trainable(m::BertEmbeddingModel) = (; trunk=m.trunk)

"""
    embed(model, input_ids::AbstractVector{<:Integer}) -> Vector

Embed one token-id sequence, returning the pooled (and, by default,
L2-normalized) sentence vector.
"""
function embed(m::BertEmbeddingModel, input_ids::AbstractVector{<:Integer}; kwargs...)
    return vec(m(reshape(collect(input_ids), :, 1); kwargs...))
end

"""
    embed(model, tokenizer, text::AbstractString) -> Vector
    embed(model, tokenizer, texts::AbstractVector{<:AbstractString}) -> Matrix

Tokenize then embed. The caller owns any model-specific input convention:
special tokens ([CLS]/[SEP]) and instruction prefixes (E5's `query:` /
`passage:`, BGE's query instruction). Each text embeds on its own, then the
columns concatenate into `(hidden, n)`; no padding mask needed.
"""
function embed(m::BertEmbeddingModel, tokenizer::Tokenizer, text::AbstractString; kwargs...)
    return embed(m, encode(tokenizer, text); kwargs...)
end

function embed(
    m::BertEmbeddingModel,
    tokenizer::Tokenizer,
    texts::AbstractVector{<:AbstractString};
    kwargs...,
)
    return reduce(hcat, (embed(m, tokenizer, t; kwargs...) for t in texts))
end

"""
    load_state_dict!(m::BertEmbeddingModel, weights) -> m

Populate the trunk in place. Reuses [`bert_state_dict_map`](@ref) filtered to
the encoder/embedding keys; it ignores any prediction-head or pooler weights in
the checkpoint.
"""
function load_state_dict!(
    m::BertEmbeddingModel, weights::AbstractDict{String,<:AbstractArray}
)
    full = bert_state_dict_map(m.config)
    trunk_map = Dict(k => v for (k, v) in full if first(first(v)) === :trunk)
    load_into!(m, weights, trunk_map)
    return m
end
