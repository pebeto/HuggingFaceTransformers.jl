using Test
using Random
using Flux
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models:
    build_caches, qwen_state_dict_map, load_state_dict!, load_into!

function _qwen_synthetic_state_dict(cfg::QwenConfig)
    out = Dict{String,Array{Float32}}()
    n_q = cfg.num_attention_heads * cfg.head_dim
    n_kv = cfg.num_key_value_heads * cfg.head_dim

    out["model.embed_tokens.weight"] = randn(Float32, cfg.vocab_size, cfg.hidden_size)
    out["model.norm.weight"] = randn(Float32, cfg.hidden_size)
    if !cfg.tie_word_embeddings
        out["lm_head.weight"] = randn(Float32, cfg.vocab_size, cfg.hidden_size)
    end

    for i in 0:(cfg.num_hidden_layers - 1)
        p = "model.layers.$(i)"
        out["$(p).input_layernorm.weight"] = randn(Float32, cfg.hidden_size)
        out["$(p).post_attention_layernorm.weight"] = randn(Float32, cfg.hidden_size)
        out["$(p).self_attn.q_proj.weight"] = randn(Float32, n_q, cfg.hidden_size)
        out["$(p).self_attn.k_proj.weight"] = randn(Float32, n_kv, cfg.hidden_size)
        out["$(p).self_attn.v_proj.weight"] = randn(Float32, n_kv, cfg.hidden_size)
        out["$(p).self_attn.o_proj.weight"] = randn(Float32, cfg.hidden_size, n_q)
        # Qwen's QKV bias (no o_proj bias).
        out["$(p).self_attn.q_proj.bias"] = randn(Float32, n_q)
        out["$(p).self_attn.k_proj.bias"] = randn(Float32, n_kv)
        out["$(p).self_attn.v_proj.bias"] = randn(Float32, n_kv)
        out["$(p).mlp.gate_proj.weight"] = randn(
            Float32, cfg.intermediate_size, cfg.hidden_size
        )
        out["$(p).mlp.up_proj.weight"] = randn(
            Float32, cfg.intermediate_size, cfg.hidden_size
        )
        out["$(p).mlp.down_proj.weight"] = randn(
            Float32, cfg.hidden_size, cfg.intermediate_size
        )
    end
    return out
end

function _tiny_qwen_config(; tie_word_embeddings=false, sliding_window=nothing)
    return QwenConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=1_000_000.0,
        rms_norm_eps=1.0e-6,
        tie_word_embeddings=tie_word_embeddings,
        sliding_window=sliding_window,
    )
end

@testset verbose = true "QwenForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_qwen_config()
    lm = QwenForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers
    @test size(lm.lm_head.weight) == (cfg.vocab_size, cfg.hidden_size)

    # Q/K/V projections must carry bias; O projection must not.
    @test all(layer.self_attn.wq.bias isa AbstractVector for layer in lm.model.layers)
    @test all(layer.self_attn.wk.bias isa AbstractVector for layer in lm.model.layers)
    @test all(layer.self_attn.wv.bias isa AbstractVector for layer in lm.model.layers)
    @test all(layer.self_attn.wo.bias === nothing for layer in lm.model.layers)

    @testset "forward shapes" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        logits = lm(ids)
        @test size(logits) == (cfg.vocab_size, 5, 2)
    end

    @testset "KV-cache prefill→decode parity" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        out_full = lm(ids)
        caches = build_caches(lm, cfg.max_position_embeddings, 1)
        out_prefill = lm(ids[1:3, :]; caches=caches, step=1)
        @test size(out_prefill) == (cfg.vocab_size, 3, 1)
        out_decode = lm(ids[4:4, :]; caches=caches, step=4)
        @test isapprox(out_full[:, 4:4, :], out_decode; atol=1.0e-4)
    end
end

@testset "qwen_state_dict_map" begin
    cfg = _tiny_qwen_config()
    sd_map = qwen_state_dict_map(cfg)

    @test haskey(sd_map, "model.embed_tokens.weight")
    @test haskey(sd_map, "model.norm.weight")
    @test haskey(sd_map, "lm_head.weight")
    @test haskey(sd_map, "model.layers.0.self_attn.q_proj.weight")
    @test haskey(sd_map, "model.layers.0.self_attn.q_proj.bias")
    @test haskey(sd_map, "model.layers.0.self_attn.k_proj.bias")
    @test haskey(sd_map, "model.layers.0.self_attn.v_proj.bias")
    # o_proj has no bias entry.
    @test !haskey(sd_map, "model.layers.0.self_attn.o_proj.bias")

    # Count: 2 top + (9 weights + 3 biases) per layer + 1 lm_head
    @test length(sd_map) == 2 + cfg.num_hidden_layers * 12 + 1

    @testset "tied embeddings drop lm_head.weight" begin
        tied_cfg = _tiny_qwen_config(; tie_word_embeddings=true)
        tied_map = qwen_state_dict_map(tied_cfg)
        @test !haskey(tied_map, "lm_head.weight")
    end
end

@testset "load_state_dict! — Qwen untied (bias loads end-to-end)" begin
    Random.seed!(0x9999)
    cfg = _tiny_qwen_config()
    lm = QwenForCausalLM(cfg)
    sd = _qwen_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    # Weights flow.
    @test lm.model.layers[1].self_attn.wq.weight ==
        sd["model.layers.0.self_attn.q_proj.weight"]
    @test lm.model.layers[2].self_attn.wo.weight ==
        sd["model.layers.1.self_attn.o_proj.weight"]
    # Biases flow into the right place.
    @test lm.model.layers[1].self_attn.wq.bias == sd["model.layers.0.self_attn.q_proj.bias"]
    @test lm.model.layers[1].self_attn.wk.bias == sd["model.layers.0.self_attn.k_proj.bias"]
    @test lm.model.layers[2].self_attn.wv.bias == sd["model.layers.1.self_attn.v_proj.bias"]
    @test lm.lm_head.weight == sd["lm_head.weight"]
end

@testset "load_state_dict! — Qwen tied embeddings" begin
    Random.seed!(0xAAAA)
    cfg = _tiny_qwen_config(; tie_word_embeddings=true)
    lm = QwenForCausalLM(cfg)
    sd = _qwen_synthetic_state_dict(cfg)
    @test !haskey(sd, "lm_head.weight")

    load_state_dict!(lm, sd)
    @test lm.lm_head.weight == sd["model.embed_tokens.weight"]
end

@testset "QKV bias actually affects the forward pass" begin
    # Build two QwenForCausalLM with same weights but different biases.
    Random.seed!(0xBEEF)
    cfg = _tiny_qwen_config()
    lm1 = QwenForCausalLM(cfg)
    lm2 = QwenForCausalLM(cfg)
    sd1 = _qwen_synthetic_state_dict(cfg)
    sd2 = deepcopy(sd1)
    # Perturb one bias entry; everything else identical.
    sd2["model.layers.0.self_attn.q_proj.bias"] .+= 1.0f0

    load_state_dict!(lm1, sd1)
    load_state_dict!(lm2, sd2)

    ids = rand(0:(cfg.vocab_size - 1), 3, 1)
    out1 = lm1(ids)
    out2 = lm2(ids)
    @test !(out1 ≈ out2)
end
