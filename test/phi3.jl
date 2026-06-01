using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: build_caches, phi3_state_dict_map, load_state_dict!, load_into!

function _phi3_synthetic_state_dict(cfg::Phi3Config)
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
        # Fused QKV: shape (n_q + 2*n_kv, hidden_size) — concatenated [Q; K; V].
        out["$(p).self_attn.qkv_proj.weight"] = randn(
            Float32, n_q + 2 * n_kv, cfg.hidden_size
        )
        out["$(p).self_attn.o_proj.weight"] = randn(Float32, cfg.hidden_size, n_q)
        # Fused gate-up: shape (2 * intermediate_size, hidden_size) — [gate; up].
        out["$(p).mlp.gate_up_proj.weight"] = randn(
            Float32, 2 * cfg.intermediate_size, cfg.hidden_size
        )
        out["$(p).mlp.down_proj.weight"] = randn(
            Float32, cfg.hidden_size, cfg.intermediate_size
        )
    end
    return out
end

function _tiny_phi3_config(; sliding_window=nothing, tie_word_embeddings=false)
    return Phi3Config(;
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

@testset verbose = true "Phi3ForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_phi3_config(; sliding_window=3)
    lm = Phi3ForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers
    @test size(lm.lm_head.weight) == (cfg.vocab_size, cfg.hidden_size)
    # Sliding window threads through to every GQA layer.
    @test all(layer.self_attn.window_size == 3 for layer in lm.model.layers)

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

@testset "phi3_state_dict_map" begin
    cfg = _tiny_phi3_config()
    sd_map = phi3_state_dict_map(cfg)

    # Standard 1:1 entries are present.
    @test haskey(sd_map, "model.embed_tokens.weight")
    @test haskey(sd_map, "model.norm.weight")
    @test haskey(sd_map, "lm_head.weight")
    @test haskey(sd_map, "model.layers.0.input_layernorm.weight")
    @test haskey(sd_map, "model.layers.0.post_attention_layernorm.weight")
    @test haskey(sd_map, "model.layers.0.self_attn.o_proj.weight")
    @test haskey(sd_map, "model.layers.0.mlp.down_proj.weight")

    # Fused weights are NOT in the map — handled by load_state_dict! itself.
    @test !haskey(sd_map, "model.layers.0.self_attn.qkv_proj.weight")
    @test !haskey(sd_map, "model.layers.0.mlp.gate_up_proj.weight")

    # 2 top + (4 per layer) + 1 lm_head
    @test length(sd_map) == 2 + cfg.num_hidden_layers * 4 + 1
end

@testset "load_state_dict! — Phi-3 slices fused QKV and gate-up" begin
    Random.seed!(0x9999)
    cfg = _tiny_phi3_config()
    lm = Phi3ForCausalLM(cfg)
    sd = _phi3_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    n_q = cfg.num_attention_heads * cfg.head_dim
    n_kv = cfg.num_key_value_heads * cfg.head_dim
    inter = cfg.intermediate_size

    # Layer 0: verify slicing by reproducing the slices from the source.
    qkv_src = sd["model.layers.0.self_attn.qkv_proj.weight"]
    @test lm.model.layers[1].self_attn.wq.weight == qkv_src[1:n_q, :]
    @test lm.model.layers[1].self_attn.wk.weight == qkv_src[(n_q + 1):(n_q + n_kv), :]
    @test lm.model.layers[1].self_attn.wv.weight ==
        qkv_src[(n_q + n_kv + 1):(n_q + 2 * n_kv), :]

    gup_src = sd["model.layers.0.mlp.gate_up_proj.weight"]
    @test lm.model.layers[1].mlp.gate_proj.weight == gup_src[1:inter, :]
    @test lm.model.layers[1].mlp.up_proj.weight == gup_src[(inter + 1):(2 * inter), :]

    # And the non-fused entries flowed too.
    @test lm.model.layers[1].self_attn.wo.weight ==
        sd["model.layers.0.self_attn.o_proj.weight"]
    @test lm.lm_head.weight == sd["lm_head.weight"]
end

@testset "load_state_dict! — Phi-3 errors on shape mismatch" begin
    cfg = _tiny_phi3_config()
    lm = Phi3ForCausalLM(cfg)
    sd = _phi3_synthetic_state_dict(cfg)
    # Corrupt one fused tensor's first dim.
    sd["model.layers.0.self_attn.qkv_proj.weight"] = randn(Float32, 7, cfg.hidden_size)
    @test_throws DimensionMismatch load_state_dict!(lm, sd)
end

@testset "load_state_dict! — Phi-3 tied embeddings" begin
    Random.seed!(0xAAAA)
    cfg = _tiny_phi3_config(; tie_word_embeddings=true)
    lm = Phi3ForCausalLM(cfg)
    sd = _phi3_synthetic_state_dict(cfg)
    @test !haskey(sd, "lm_head.weight")

    load_state_dict!(lm, sd)
    @test lm.lm_head.weight == sd["model.embed_tokens.weight"]
end
