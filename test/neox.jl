using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: build_caches, neox_state_dict_map, load_state_dict!, load_into!

function _neox_synthetic_state_dict(cfg::NeoXConfig)
    out = Dict{String,Array{Float32}}()
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    hd = cfg.head_dim

    out["gpt_neox.embed_in.weight"] = randn(Float32, cfg.vocab_size, h)
    out["gpt_neox.final_layer_norm.weight"] = randn(Float32, h)
    out["gpt_neox.final_layer_norm.bias"] = randn(Float32, h)
    if !cfg.tie_word_embeddings
        out["embed_out.weight"] = randn(Float32, cfg.vocab_size, h)
    end

    for i in 0:(cfg.num_hidden_layers - 1)
        p = "gpt_neox.layers.$(i)"
        out["$(p).input_layernorm.weight"] = randn(Float32, h)
        out["$(p).input_layernorm.bias"] = randn(Float32, h)
        out["$(p).post_attention_layernorm.weight"] = randn(Float32, h)
        out["$(p).post_attention_layernorm.bias"] = randn(Float32, h)

        # Fused QKV: (3*n_heads*hd, hidden), per-head interleaved.
        out["$(p).attention.query_key_value.weight"] = randn(Float32, 3 * n_heads * hd, h)
        out["$(p).attention.query_key_value.bias"] = randn(Float32, 3 * n_heads * hd)

        # Output projection (already in standard (out, in) — NeoX uses nn.Linear,
        # not Conv1D, so no transpose at load time). Shape is (hidden, n_heads*hd);
        # GQA's `wo = Linear(n_heads*hd, hidden)` stores it as (hidden, n_heads*hd).
        out["$(p).attention.dense.weight"] = randn(Float32, h, n_heads * hd)
        out["$(p).attention.dense.bias"] = randn(Float32, h)

        out["$(p).mlp.dense_h_to_4h.weight"] = randn(Float32, inter, h)
        out["$(p).mlp.dense_h_to_4h.bias"] = randn(Float32, inter)
        out["$(p).mlp.dense_4h_to_h.weight"] = randn(Float32, h, inter)
        out["$(p).mlp.dense_4h_to_h.bias"] = randn(Float32, h)
    end
    return out
end

function _tiny_neox_config(; partial_rotary_factor=0.25)
    # hidden_size = num_attention_heads * head_dim (= 32) so wo's shape
    # matches naturally; head_dim=8 makes rotary_dim=2 (even) under 0.25.
    return NeoXConfig(;
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        max_position_embeddings=16,
        rope_theta=10_000.0,
        partial_rotary_factor=partial_rotary_factor,
        layer_norm_eps=1.0e-5,
        tie_word_embeddings=false,
    )
end

@testset verbose = true "NeoXForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_neox_config()
    lm = NeoXForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers
    @test size(lm.lm_head.weight) == (cfg.vocab_size, cfg.hidden_size)

    @testset "layer kit composition" begin
        for layer in lm.model.layers
            @test layer isa Allspark.Models.NeoXDecoderLayer
            @test layer.input_layernorm isa Allspark.Layers.LayerNorm
            @test layer.post_attention_layernorm isa Allspark.Layers.LayerNorm
            @test layer.mlp isa Allspark.Layers.GeluMLP
            # MHA: kv heads == q heads.
            @test layer.self_attn.num_heads_q == layer.self_attn.num_heads_k
            # Biases on Q/K/V and O.
            @test layer.self_attn.wq.bias isa AbstractVector
            @test layer.self_attn.wo.bias isa AbstractVector
        end
        # Partial rotary: rope was built with rotary_dim = head_dim * 0.25 = 2.
        @test lm.model.layers[1].self_attn.rope.rotary_dim == 2
    end

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

@testset "NeoX rejects odd partial-rotary dim" begin
    # head_dim=4 * 0.25 = 1; rotation operates on pairs so it must be even.
    cfg = NeoXConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=4,                 # → rotary_dim=1, invalid
        max_position_embeddings=16,
        partial_rotary_factor=0.25,
    )
    @test_throws ArgumentError NeoXForCausalLM(cfg)
end

@testset "neox_state_dict_map" begin
    cfg = _tiny_neox_config()
    sd_map = neox_state_dict_map(cfg)

    # Trunk lives under `gpt_neox.…`.
    @test haskey(sd_map, "gpt_neox.embed_in.weight")
    @test haskey(sd_map, "gpt_neox.final_layer_norm.weight")
    @test haskey(sd_map, "gpt_neox.layers.0.input_layernorm.weight")
    @test haskey(sd_map, "gpt_neox.layers.0.attention.dense.weight")
    @test haskey(sd_map, "gpt_neox.layers.0.mlp.dense_h_to_4h.weight")
    # LM head lives at the root, not under `gpt_neox.`.
    @test haskey(sd_map, "embed_out.weight")

    # Fused QKV is NOT in the map — handled directly in load_state_dict!.
    @test !haskey(sd_map, "gpt_neox.layers.0.attention.query_key_value.weight")
    @test !haskey(sd_map, "gpt_neox.layers.0.attention.query_key_value.bias")

    # embed_in transposes; everything else is as_is (NeoX uses nn.Linear).
    @test sd_map["gpt_neox.embed_in.weight"][2] == :transpose
    @test sd_map["gpt_neox.layers.0.attention.dense.weight"][2] == :as_is
    @test sd_map["embed_out.weight"][2] == :as_is

    # 1 embed + 2 final norm + (10 entries per layer) + 1 lm_head
    @test length(sd_map) == 3 + cfg.num_hidden_layers * 10 + 1
end

@testset "load_state_dict! — NeoX slices interleaved fused QKV per head" begin
    Random.seed!(0x9999)
    cfg = _tiny_neox_config()
    lm = NeoXForCausalLM(cfg)
    sd = _neox_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    # Spot-check 1:1 entries flowed.
    @test lm.model.embed_tokens.weight ==
        permutedims(sd["gpt_neox.embed_in.weight"], (2, 1))
    @test lm.model.norm.bias == sd["gpt_neox.final_layer_norm.bias"]
    @test lm.model.layers[1].self_attn.wo.weight ==
        sd["gpt_neox.layers.0.attention.dense.weight"]
    @test lm.lm_head.weight == sd["embed_out.weight"]   # no transpose on embed_out

    # Per-head interleaved slicing: reproduce one head's slice from the source.
    qkv_w = sd["gpt_neox.layers.0.attention.query_key_value.weight"]
    qkv_b = sd["gpt_neox.layers.0.attention.query_key_value.bias"]
    hd = cfg.head_dim
    for head in 0:(cfg.num_attention_heads - 1)
        src_base = head * 3 * hd
        dst_rows = (head * hd + 1):((head + 1) * hd)

        @test lm.model.layers[1].self_attn.wq.weight[dst_rows, :] ==
            qkv_w[(src_base + 1):(src_base + hd), :]
        @test lm.model.layers[1].self_attn.wk.weight[dst_rows, :] ==
            qkv_w[(src_base + hd + 1):(src_base + 2 * hd), :]
        @test lm.model.layers[1].self_attn.wv.weight[dst_rows, :] ==
            qkv_w[(src_base + 2 * hd + 1):(src_base + 3 * hd), :]

        @test lm.model.layers[1].self_attn.wq.bias[dst_rows] ==
            qkv_b[(src_base + 1):(src_base + hd)]
        @test lm.model.layers[1].self_attn.wk.bias[dst_rows] ==
            qkv_b[(src_base + hd + 1):(src_base + 2 * hd)]
        @test lm.model.layers[1].self_attn.wv.bias[dst_rows] ==
            qkv_b[(src_base + 2 * hd + 1):(src_base + 3 * hd)]
    end
end

@testset "load_state_dict! — NeoX errors on fused-QKV shape mismatch" begin
    cfg = _tiny_neox_config()
    lm = NeoXForCausalLM(cfg)
    sd = _neox_synthetic_state_dict(cfg)
    sd["gpt_neox.layers.0.attention.query_key_value.weight"] = randn(
        Float32, 5, cfg.hidden_size
    )   # wrong first dim
    @test_throws DimensionMismatch load_state_dict!(lm, sd)
end

@testset "NeoXDecoderLayer is parallel-residual" begin
    # Compare a NeoXDecoderLayer's output against the manual parallel-residual
    # computation. If the layer accidentally became sequential, the result
    # would diverge.
    Random.seed!(0xBEEF)
    cfg = _tiny_neox_config()
    lm = NeoXForCausalLM(cfg)
    sd = _neox_synthetic_state_dict(cfg)
    load_state_dict!(lm, sd)

    layer = lm.model.layers[1]
    x = randn(Float32, cfg.hidden_size, 3, 1)

    expected = let
        h_attn = layer.input_layernorm(x)
        h_attn = layer.self_attn(h_attn; position_ids=collect(0:2))
        h_mlp = layer.post_attention_layernorm(x)        # NOTE: from x, not x+h_attn
        h_mlp = layer.mlp(h_mlp)
        x .+ h_attn .+ h_mlp
    end

    @test layer(x; position_ids=collect(0:2)) ≈ expected
end
