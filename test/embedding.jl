using Test
using Random
using Flux
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models:
    bert_state_dict_map, load_state_dict!, mean_pool, cls_pool, l2_normalize
using HuggingFaceTransformers.Tokenizers: Tokenizer
using HuggingFaceTransformers.Tokenizers:
    BPEModel,
    ByteLevelPreTokenizer,
    ByteLevelDecoder,
    BYTES_TO_UNICODE,
    AddedToken,
    IdentityNormalizer

function _emb_cfg(; prefix="bert", offset=0, maxpos=16, vocab=64)
    return BertConfig(;
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=maxpos,
        type_vocab_size=2,
        layer_norm_eps=1.0e-12,
        hf_prefix=prefix,
        position_embedding_offset=offset,
    )
end

# Trunk-only synthetic state dict honoring hf_prefix (incl. empty).
function _emb_trunk_sd(cfg::BertConfig)
    p = cfg.hf_prefix
    pfx(s) = isempty(p) ? s : "$(p).$(s)"
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    out = Dict{String,Array{Float32}}()
    out[pfx("embeddings.word_embeddings.weight")] = randn(Float32, cfg.vocab_size, h)
    out[pfx("embeddings.position_embeddings.weight")] = randn(
        Float32, cfg.max_position_embeddings + cfg.position_embedding_offset, h
    )
    out[pfx("embeddings.token_type_embeddings.weight")] = randn(
        Float32, cfg.type_vocab_size, h
    )
    out[pfx("embeddings.LayerNorm.weight")] = randn(Float32, h)
    out[pfx("embeddings.LayerNorm.bias")] = randn(Float32, h)
    for i in 0:(cfg.num_hidden_layers - 1)
        prefix = pfx("encoder.layer.$(i)")
        out["$(prefix).attention.self.query.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.self.query.bias"] = randn(Float32, h)
        out["$(prefix).attention.self.key.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.self.key.bias"] = randn(Float32, h)
        out["$(prefix).attention.self.value.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.self.value.bias"] = randn(Float32, h)
        out["$(prefix).attention.output.dense.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.output.dense.bias"] = randn(Float32, h)
        out["$(prefix).attention.output.LayerNorm.weight"] = randn(Float32, h)
        out["$(prefix).attention.output.LayerNorm.bias"] = randn(Float32, h)
        out["$(prefix).intermediate.dense.weight"] = randn(Float32, inter, h)
        out["$(prefix).intermediate.dense.bias"] = randn(Float32, inter)
        out["$(prefix).output.dense.weight"] = randn(Float32, h, inter)
        out["$(prefix).output.dense.bias"] = randn(Float32, h)
        out["$(prefix).output.LayerNorm.weight"] = randn(Float32, h)
        out["$(prefix).output.LayerNorm.bias"] = randn(Float32, h)
    end
    return out
end

@testset "pooling primitives" begin
    h = reshape(Float32.(1:6), 2, 3, 1)        # [:, :, 1] == [1 3 5; 2 4 6]
    @test vec(mean_pool(h)) ≈ [3.0f0, 4.0f0]
    @test vec(cls_pool(h)) ≈ [1.0f0, 2.0f0]
    @test l2_normalize(Float32[3.0; 4.0;;]) ≈ Float32[0.6; 0.8;;]
    @test all(isfinite, l2_normalize(zeros(Float32, 4, 1)))   # zero column ≠ NaN
end

@testset "BertEmbeddingModel forward — shape & unit norm" begin
    Random.seed!(0xE1)
    cfg = _emb_cfg()
    m = BertEmbeddingModel(cfg; pooling=:cls, normalize=true)
    ids = rand(0:(cfg.vocab_size - 1), 5, 3)
    out = m(ids)
    @test size(out) == (cfg.hidden_size, 3)
    @test all(isapprox.(vec(sqrt.(sum(abs2, out; dims=1))), 1.0f0; atol=1.0e-5))

    # normalize=false leaves magnitude alone.
    m2 = BertEmbeddingModel(cfg; pooling=:cls, normalize=false)
    @test !all(isapprox.(vec(sqrt.(sum(abs2, m2(ids); dims=1))), 1.0f0; atol=1.0e-3))
end

@testset "pooling strategy validation" begin
    @test_throws ArgumentError BertEmbeddingModel(_emb_cfg(); pooling=:bogus)
end

@testset "embed(ids) == pooled + normalized trunk output" begin
    Random.seed!(0xE2)
    cfg = _emb_cfg()
    for pool in (:cls, :mean)
        m = BertEmbeddingModel(cfg; pooling=pool, normalize=true)
        ids = [1, 5, 9, 2]
        h = m.trunk(reshape(ids, :, 1))
        manual = l2_normalize(pool === :cls ? cls_pool(h) : mean_pool(h))
        v = embed(m, ids)
        @test v isa Vector{Float32}
        @test length(v) == cfg.hidden_size
        @test v ≈ vec(manual)
        @test sum(v .* embed(m, ids)) ≈ 1.0f0 atol = 1.0e-5   # cosine self-sim
    end
end

@testset "cls vs mean pooling diverge (shared weights)" begin
    Random.seed!(0xE3)
    cfg = _emb_cfg()
    sd = _emb_trunk_sd(cfg)
    m_cls = BertEmbeddingModel(cfg; pooling=:cls)
    m_mean = BertEmbeddingModel(cfg; pooling=:mean)
    load_state_dict!(m_cls, sd)
    load_state_dict!(m_mean, sd)
    ids = [3, 1, 4, 1, 5]
    @test embed(m_cls, ids) != embed(m_mean, ids)
    @test m_cls.trunk.embeddings.embed_tokens.weight ==
        m_mean.trunk.embeddings.embed_tokens.weight
end

@testset "load_state_dict! populates the trunk" begin
    Random.seed!(0xE4)
    cfg = _emb_cfg()
    m = BertEmbeddingModel(cfg)
    sd = _emb_trunk_sd(cfg)
    load_state_dict!(m, sd)
    @test m.trunk.embeddings.embed_tokens.weight ==
        permutedims(sd["bert.embeddings.word_embeddings.weight"], (2, 1))
    @test m.trunk.encoder.layers[1].self_attn.wq.weight ==
        sd["bert.encoder.layer.0.attention.self.query.weight"]
    @test m.trunk.encoder.layers[2].output_dense.bias ==
        sd["bert.encoder.layer.1.output.dense.bias"]
end

@testset "prefix-less checkpoint (BGE-style) loads" begin
    Random.seed!(0xE5)
    cfg = _emb_cfg(; prefix="")
    mp = bert_state_dict_map(cfg)
    @test haskey(mp, "embeddings.word_embeddings.weight")
    @test !any(startswith(k, ".") for k in keys(mp))

    m = BertEmbeddingModel(cfg)
    sd = _emb_trunk_sd(cfg)
    load_state_dict!(m, sd)
    @test m.trunk.embeddings.embed_tokens.weight ==
        permutedims(sd["embeddings.word_embeddings.weight"], (2, 1))
end

@testset "embed(model, tokenizer, text / texts)" begin
    # Byte-level tokenizer over the first 256 IDs (matches generate.jl's fixture).
    vocab = Dict{String,Int}()
    for b in 0x00:0xff
        vocab[string(BYTES_TO_UNICODE[b])] = Int(b)
    end
    tk = Tokenizer(
        BPEModel(vocab, Tuple{String,String}[]),
        IdentityNormalizer(),
        ByteLevelPreTokenizer(false, true),
        ByteLevelDecoder(),
        AddedToken[],
        Dict{String,Int}(),
        Dict{Int,String}(),
    )
    Random.seed!(0xE6)
    cfg = _emb_cfg(; vocab=256, maxpos=64)
    m = BertEmbeddingModel(cfg; pooling=:mean)

    v = embed(m, tk, "hello")
    @test v isa Vector{Float32}
    @test length(v) == cfg.hidden_size

    M = embed(m, tk, ["hello", "world!"])
    @test size(M) == (cfg.hidden_size, 2)
end
