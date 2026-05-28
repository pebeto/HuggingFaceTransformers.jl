using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: build_caches, mistral_state_dict_map, load_state_dict!, load_into!

function _mistral_synthetic_state_dict(cfg::MistralConfig)
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
        out["$(p).mlp.gate_proj.weight"] =
            randn(Float32, cfg.intermediate_size, cfg.hidden_size)
        out["$(p).mlp.up_proj.weight"] =
            randn(Float32, cfg.intermediate_size, cfg.hidden_size)
        out["$(p).mlp.down_proj.weight"] =
            randn(Float32, cfg.hidden_size, cfg.intermediate_size)
    end
    return out
end

function _tiny_mistral_config(; sliding_window=nothing, tie_word_embeddings=false)
    return MistralConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=10_000.0,
        rms_norm_eps=1.0e-5,
        tie_word_embeddings=tie_word_embeddings,
        sliding_window=sliding_window,
    )
end

@testset verbose = true "MistralForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_mistral_config(; sliding_window=3)
    lm = MistralForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers
    @test size(lm.lm_head.weight) == (cfg.vocab_size, cfg.hidden_size)
    # Sliding window must be threaded through to every GQA.
    @test all(layer.self_attn.window_size == 3 for layer in lm.model.layers)

    @testset "forward shapes" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        logits = lm(ids)
        @test size(logits) == (cfg.vocab_size, 5, 2)
    end

    @testset "KV-cache prefill→decode parity (with sliding window)" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        out_full = lm(ids)

        caches = build_caches(lm, cfg.max_position_embeddings, 1)
        out_prefill = lm(ids[1:3, :]; caches=caches, step=1)
        @test size(out_prefill) == (cfg.vocab_size, 3, 1)

        out_decode = lm(ids[4:4, :]; caches=caches, step=4)
        @test size(out_decode) == (cfg.vocab_size, 1, 1)

        @test isapprox(out_full[:, 4:4, :], out_decode; atol=1.0e-4)
    end
end

@testset "mistral_state_dict_map" begin
    cfg = _tiny_mistral_config()
    sd_map = mistral_state_dict_map(cfg)

    # Llama and Mistral share the HF parameter naming convention.
    @test haskey(sd_map, "model.embed_tokens.weight")
    @test haskey(sd_map, "model.norm.weight")
    @test haskey(sd_map, "lm_head.weight")
    @test haskey(sd_map, "model.layers.0.self_attn.q_proj.weight")
    @test haskey(sd_map, "model.layers.0.mlp.gate_proj.weight")
    @test haskey(sd_map, "model.layers.1.self_attn.o_proj.weight")
    @test length(sd_map) == 2 + cfg.num_hidden_layers * 9 + 1  # embed + norm + 9/layer + lm_head

    @testset "tied embeddings drop lm_head.weight" begin
        tied_cfg = _tiny_mistral_config(; tie_word_embeddings=true)
        tied_map = mistral_state_dict_map(tied_cfg)
        @test !haskey(tied_map, "lm_head.weight")
    end
end

@testset "load_state_dict! — Mistral untied" begin
    Random.seed!(0x9999)
    cfg = _tiny_mistral_config()
    lm = MistralForCausalLM(cfg)
    sd = _mistral_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    # Spot-check that values flowed end-to-end.
    @test lm.model.norm.weight == sd["model.norm.weight"]
    @test lm.model.layers[1].self_attn.wq.weight == sd["model.layers.0.self_attn.q_proj.weight"]
    @test lm.model.layers[2].mlp.down_proj.weight == sd["model.layers.1.mlp.down_proj.weight"]
    @test lm.lm_head.weight == sd["lm_head.weight"]
end

@testset "load_state_dict! — Mistral tied embeddings" begin
    Random.seed!(0xAAAA)
    cfg = _tiny_mistral_config(; tie_word_embeddings=true)
    lm = MistralForCausalLM(cfg)
    sd = _mistral_synthetic_state_dict(cfg)
    @test !haskey(sd, "lm_head.weight")

    load_state_dict!(lm, sd)
    # lm_head.weight is the transpose of the input embedding tensor when tied.
    @test lm.lm_head.weight == sd["model.embed_tokens.weight"]
end

@testset "sliding window changes attention beyond the window" begin
    # Build two configs with identical everything except window. Share weights
    # by loading the same synthetic state dict into both.
    Random.seed!(0xBEEF)
    cfg_full = _tiny_mistral_config()                              # no window
    cfg_win = _tiny_mistral_config(; sliding_window=2)             # short window

    lm_full = MistralForCausalLM(cfg_full)
    lm_win = MistralForCausalLM(cfg_win)

    sd = _mistral_synthetic_state_dict(cfg_full)
    load_state_dict!(lm_full, sd)
    load_state_dict!(lm_win, sd)

    # Sequence longer than the window. Beyond position window-1, results
    # must diverge: the windowed model can't see the earliest tokens.
    ids = rand(0:(cfg_full.vocab_size - 1), 5, 1)
    out_full = lm_full(ids)
    out_win = lm_win(ids)

    @test out_full[:, 1, 1] ≈ out_win[:, 1, 1]              # pos 0: same context (only self)
    @test out_full[:, 2, 1] ≈ out_win[:, 2, 1]              # pos 1: both see {0,1}
    @test !(out_full[:, 5, 1] ≈ out_win[:, 5, 1])           # pos 4: full sees {0..4}, win sees {3,4}
end
