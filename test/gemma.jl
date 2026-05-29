using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: build_caches, gemma_state_dict_map, load_state_dict!, load_into!

function _gemma_synthetic_state_dict(cfg::GemmaConfig)
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
        out["$(p).pre_feedforward_layernorm.weight"] = randn(Float32, cfg.hidden_size)
        out["$(p).post_feedforward_layernorm.weight"] = randn(Float32, cfg.hidden_size)
        out["$(p).self_attn.q_proj.weight"] = randn(Float32, n_q, cfg.hidden_size)
        out["$(p).self_attn.k_proj.weight"] = randn(Float32, n_kv, cfg.hidden_size)
        out["$(p).self_attn.v_proj.weight"] = randn(Float32, n_kv, cfg.hidden_size)
        out["$(p).self_attn.o_proj.weight"] = randn(Float32, cfg.hidden_size, n_q)
        out["$(p).mlp.gate_proj.weight"] =
            randn(Float32, cfg.intermediate_size, cfg.hidden_size)
        out["$(p).mlp.up_proj.weight"] =
            randn(Float32, cfg.intermediate_size, cfg.hidden_size)
        out["$(p).mlp.down_proj.weight"] =
            randn(Float32, cfg.hidden_size, cfg.intermediate_size)
    end
    return out
end

function _tiny_gemma_config(;
    sliding_window=nothing,
    attn_softcap=nothing,
    final_softcap=nothing,
    query_pre_attn_scalar=nothing,
)
    return GemmaConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=10_000.0,
        rms_norm_eps=1.0e-6,
        tie_word_embeddings=true,
        sliding_window=sliding_window,
        attn_logit_softcapping=attn_softcap,
        final_logit_softcapping=final_softcap,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )
end

@testset verbose = true "GemmaForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_gemma_config(;
        sliding_window=3, attn_softcap=50.0, final_softcap=30.0
    )
    lm = GemmaForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers

    @testset "alternating sliding-window: even layers windowed, odd full causal" begin
        # Layer indices in Julia are 1-based; HF layer 0 is Julia layer 1.
        # HF i=0 (even) → sliding; HF i=1 (odd) → full; etc.
        @test lm.model.layers[1].self_attn.window_size == 3   # i=0
        @test lm.model.layers[2].self_attn.window_size === nothing  # i=1
        @test lm.model.layers[3].self_attn.window_size == 3   # i=2
        @test lm.model.layers[4].self_attn.window_size === nothing  # i=3
    end

    @testset "attention softcap is wired into every GQA" begin
        for layer in lm.model.layers
            @test layer.self_attn.softcap == 50.0f0
        end
    end

    @testset "forward shapes" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        logits = lm(ids)
        @test size(logits) == (cfg.vocab_size, 5, 2)
    end

    @testset "final logit softcap bounds the output" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 1)
        logits = lm(ids)
        # Final softcap = 30 → all logits strictly inside (-30, 30).
        @test all(abs.(logits) .< 30.0f0)
    end

    @testset "KV-cache prefill→decode parity (with alternating windows)" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        out_full = lm(ids)
        caches = build_caches(lm, cfg.max_position_embeddings, 1)
        out_prefill = lm(ids[1:3, :]; caches=caches, step=1)
        @test size(out_prefill) == (cfg.vocab_size, 3, 1)
        out_decode = lm(ids[4:4, :]; caches=caches, step=4)
        @test isapprox(out_full[:, 4:4, :], out_decode; atol=1.0e-4)
    end
end

@testset "GemmaDecoderLayer has 4 RMSNorms" begin
    cfg = _tiny_gemma_config()
    lm = GemmaForCausalLM(cfg)
    layer = lm.model.layers[1]
    # Spot-check the four norm fields exist and are Gemma's variant.
    @test layer.input_layernorm isa Allspark.Layers.GemmaRMSNorm
    @test layer.post_attention_layernorm isa Allspark.Layers.GemmaRMSNorm
    @test layer.pre_feedforward_layernorm isa Allspark.Layers.GemmaRMSNorm
    @test layer.post_feedforward_layernorm isa Allspark.Layers.GemmaRMSNorm
end

@testset "gemma_state_dict_map" begin
    cfg = _tiny_gemma_config()
    sd_map = gemma_state_dict_map(cfg)

    @test haskey(sd_map, "model.embed_tokens.weight")
    @test haskey(sd_map, "model.norm.weight")
    @test haskey(sd_map, "model.layers.0.input_layernorm.weight")
    @test haskey(sd_map, "model.layers.0.post_attention_layernorm.weight")
    @test haskey(sd_map, "model.layers.0.pre_feedforward_layernorm.weight")
    @test haskey(sd_map, "model.layers.0.post_feedforward_layernorm.weight")
    @test haskey(sd_map, "model.layers.0.self_attn.q_proj.weight")
    @test haskey(sd_map, "model.layers.0.mlp.gate_proj.weight")

    # Tied embeddings (Gemma2 default) → no lm_head.weight entry.
    @test !haskey(sd_map, "lm_head.weight")

    # 2 top entries + (8 weight + 4 norm = 12 entries per layer)
    @test length(sd_map) == 2 + cfg.num_hidden_layers * 11
end

@testset "load_state_dict! — Gemma (always tied)" begin
    Random.seed!(0x9999)
    cfg = _tiny_gemma_config()
    lm = GemmaForCausalLM(cfg)
    sd = _gemma_synthetic_state_dict(cfg)
    @test !haskey(sd, "lm_head.weight")

    load_state_dict!(lm, sd)

    @test lm.model.layers[1].self_attn.wq.weight == sd["model.layers.0.self_attn.q_proj.weight"]
    @test lm.model.layers[2].pre_feedforward_layernorm.weight ==
        sd["model.layers.1.pre_feedforward_layernorm.weight"]
    @test lm.model.layers[3].post_feedforward_layernorm.weight ==
        sd["model.layers.2.post_feedforward_layernorm.weight"]
    @test lm.lm_head.weight == sd["model.embed_tokens.weight"]   # tied
end

@testset "Gemma2-2B-like query_pre_attn_scalar (= 224, distinct from head_dim)" begin
    cfg = GemmaConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        query_pre_attn_scalar=224,            # 2B-specific
    )
    lm = GemmaForCausalLM(cfg)
    expected_scale = sqrt(Float32(224))
    @test all(layer.self_attn.query_scale ≈ expected_scale for layer in lm.model.layers)
end
