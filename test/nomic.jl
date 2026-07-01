using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: nomic_state_dict_map, load_state_dict!

function _tiny_nomic_config(; vocab=64)
    return NomicBertConfig(;
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=4,
        type_vocab_size=2,
        layer_norm_eps=1.0e-12,
        rope_theta=1000.0,
    )
end

function _nomic_synthetic_sd(cfg::NomicBertConfig)
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    out = Dict{String,Array{Float32}}()
    out["embeddings.word_embeddings.weight"] = randn(Float32, cfg.vocab_size, h)
    out["embeddings.token_type_embeddings.weight"] = randn(Float32, cfg.type_vocab_size, h)
    out["emb_ln.weight"] = randn(Float32, h)
    out["emb_ln.bias"] = randn(Float32, h)
    for i in 0:(cfg.num_hidden_layers - 1)
        p = "encoder.layers.$(i)"
        out["$(p).norm1.weight"] = randn(Float32, h)
        out["$(p).norm1.bias"] = randn(Float32, h)
        out["$(p).attn.Wqkv.weight"] = randn(Float32, 3h, h)   # [Q; K; V]
        out["$(p).attn.out_proj.weight"] = randn(Float32, h, h)
        out["$(p).mlp.fc11.weight"] = randn(Float32, inter, h)
        out["$(p).mlp.fc12.weight"] = randn(Float32, inter, h)
        out["$(p).mlp.fc2.weight"] = randn(Float32, h, inter)
        out["$(p).norm2.weight"] = randn(Float32, h)
        out["$(p).norm2.bias"] = randn(Float32, h)
    end
    return out
end

@testset "NomicBertModel — architecture" begin
    Random.seed!(0xA1)
    cfg = _tiny_nomic_config()
    trunk = NomicBertModel(cfg)

    @test length(trunk.layers) == cfg.num_hidden_layers
    # No absolute position table; rotary lives in attention.
    @test fieldnames(typeof(trunk.embeddings)) == (:embed_tokens, :embed_types, :norm)
    for layer in trunk.layers
        @test layer.self_attn.causal == false                 # bidirectional
        @test layer.self_attn.rope isa Allspark.Layers.RoPE   # rotary present
    end

    @testset "forward shape" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        h = trunk(ids)
        @test size(h) == (cfg.hidden_size, 5, 2)
    end
end

@testset "load_state_dict! splits fused Wqkv and maps SwiGLU" begin
    Random.seed!(0xA2)
    cfg = _tiny_nomic_config()
    m = BertEmbeddingModel(cfg)                 # pooling defaults to :mean
    @test m.pooling === :mean
    sd = _nomic_synthetic_sd(cfg)
    load_state_dict!(m, sd)

    h = cfg.hidden_size
    attn = m.trunk.layers[1].self_attn
    W = sd["encoder.layers.0.attn.Wqkv.weight"]
    @test attn.wq.weight == W[1:h, :]
    @test attn.wk.weight == W[(h + 1):(2h), :]
    @test attn.wv.weight == W[(2h + 1):(3h), :]
    @test attn.wo.weight == sd["encoder.layers.0.attn.out_proj.weight"]

    # SwiGLU mapping: fc12 → gate, fc11 → up, fc2 → down.
    mlp = m.trunk.layers[1].mlp
    @test mlp.gate_proj.weight == sd["encoder.layers.0.mlp.fc12.weight"]
    @test mlp.up_proj.weight == sd["encoder.layers.0.mlp.fc11.weight"]
    @test mlp.down_proj.weight == sd["encoder.layers.0.mlp.fc2.weight"]

    # Embeddings transpose; emb_ln as-is.
    @test m.trunk.embeddings.embed_tokens.weight ==
        permutedims(sd["embeddings.word_embeddings.weight"], (2, 1))
    @test m.trunk.embeddings.norm.weight == sd["emb_ln.weight"]
    @test m.trunk.layers[2].norm2.bias == sd["encoder.layers.1.norm2.bias"]
end

@testset "nomic embedding model — mean pooled, unit norm" begin
    Random.seed!(0xA3)
    cfg = _tiny_nomic_config()
    m = BertEmbeddingModel(cfg)
    load_state_dict!(m, _nomic_synthetic_sd(cfg))

    ids = [1, 5, 9, 2, 7]
    v = embed(m, ids)
    @test v isa Vector{Float32}
    @test length(v) == cfg.hidden_size
    @test sqrt(sum(abs2, v)) ≈ 1.0f0 atol = 1.0e-5

    # embed(ids) equals a manual mean-pool + normalize of the trunk output.
    hs = m.trunk(reshape(ids, :, 1))
    manual = Allspark.Models.l2_normalize(Allspark.Models.mean_pool(hs))
    @test v ≈ vec(manual)
end

@testset "nomic_state_dict_map omits the fused Wqkv" begin
    cfg = _tiny_nomic_config()
    mp = nomic_state_dict_map(cfg)
    @test haskey(mp, "encoder.layers.0.attn.out_proj.weight")
    @test haskey(mp, "encoder.layers.0.mlp.fc11.weight")
    @test !haskey(mp, "encoder.layers.0.attn.Wqkv.weight")   # split by hand
    @test mp["embeddings.word_embeddings.weight"][2] == :transpose
end

@testset "nomic pooling validation" begin
    @test_throws ArgumentError BertEmbeddingModel(_tiny_nomic_config(); pooling=:bogus)
end
