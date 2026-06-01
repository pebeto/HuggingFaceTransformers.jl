using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: build_caches, gpt2_state_dict_map, load_state_dict!, load_into!

function _gpt2_synthetic_state_dict(cfg::GPT2Config)
    out = Dict{String,Array{Float32}}()

    # HF Conv1D orientation: (in, out) for weights, (out,) for biases.
    h = cfg.hidden_size
    inter = cfg.intermediate_size

    out["transformer.wte.weight"] = randn(Float32, cfg.vocab_size, h)
    out["transformer.wpe.weight"] = randn(Float32, cfg.max_position_embeddings, h)
    out["transformer.ln_f.weight"] = randn(Float32, h)
    out["transformer.ln_f.bias"] = randn(Float32, h)

    for i in 0:(cfg.num_hidden_layers - 1)
        p = "transformer.h.$(i)"
        out["$(p).ln_1.weight"] = randn(Float32, h)
        out["$(p).ln_1.bias"] = randn(Float32, h)
        out["$(p).ln_2.weight"] = randn(Float32, h)
        out["$(p).ln_2.bias"] = randn(Float32, h)
        # Fused QKV: Conv1D (hidden, 3*hidden), bias (3*hidden,).
        out["$(p).attn.c_attn.weight"] = randn(Float32, h, 3 * h)
        out["$(p).attn.c_attn.bias"] = randn(Float32, 3 * h)
        out["$(p).attn.c_proj.weight"] = randn(Float32, h, h)
        out["$(p).attn.c_proj.bias"] = randn(Float32, h)
        out["$(p).mlp.c_fc.weight"] = randn(Float32, h, inter)
        out["$(p).mlp.c_fc.bias"] = randn(Float32, inter)
        out["$(p).mlp.c_proj.weight"] = randn(Float32, inter, h)
        out["$(p).mlp.c_proj.bias"] = randn(Float32, h)
    end
    return out
end

function _tiny_gpt2_config()
    return GPT2Config(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        layer_norm_eps=1.0e-5,
        tie_word_embeddings=true,
    )
end

@testset verbose = true "GPT2ForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_gpt2_config()
    lm = GPT2ForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers
    @test size(lm.lm_head.weight) == (cfg.vocab_size, cfg.hidden_size)

    @testset "no RoPE; QKV+O all biased" begin
        for layer in lm.model.layers
            @test layer.self_attn.rope === nothing
            @test layer.self_attn.wq.bias isa AbstractVector
            @test layer.self_attn.wk.bias isa AbstractVector
            @test layer.self_attn.wv.bias isa AbstractVector
            @test layer.self_attn.wo.bias isa AbstractVector
        end
    end

    @testset "decoder block uses LayerNorm (not RMSNorm)" begin
        for layer in lm.model.layers
            @test layer.input_layernorm isa Allspark.Layers.LayerNorm
            @test layer.post_attention_layernorm isa Allspark.Layers.LayerNorm
            @test layer.mlp isa Allspark.Layers.GeluMLP
        end
        @test lm.model.norm isa Allspark.Layers.LayerNorm
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

@testset "gpt2_state_dict_map" begin
    cfg = _tiny_gpt2_config()
    sd_map = gpt2_state_dict_map(cfg)

    @test haskey(sd_map, "transformer.wte.weight")
    @test haskey(sd_map, "transformer.wpe.weight")
    @test haskey(sd_map, "transformer.ln_f.weight")
    @test haskey(sd_map, "transformer.ln_f.bias")
    @test haskey(sd_map, "transformer.h.0.ln_1.weight")
    @test haskey(sd_map, "transformer.h.0.ln_1.bias")
    @test haskey(sd_map, "transformer.h.0.ln_2.weight")
    @test haskey(sd_map, "transformer.h.0.ln_2.bias")

    # wte / wpe transpose on load (HF stores (vocab, hidden); we want (hidden, vocab)).
    @test sd_map["transformer.wte.weight"][2] == :transpose
    @test sd_map["transformer.wpe.weight"][2] == :transpose

    # Fused / Conv1D entries are NOT in the map — handled by load_state_dict! directly.
    @test !haskey(sd_map, "transformer.h.0.attn.c_attn.weight")
    @test !haskey(sd_map, "transformer.h.0.attn.c_proj.weight")
    @test !haskey(sd_map, "transformer.h.0.mlp.c_fc.weight")
    @test !haskey(sd_map, "transformer.h.0.mlp.c_proj.weight")
    # Tied embeddings: no lm_head.weight entry either.
    @test !haskey(sd_map, "lm_head.weight")

    # 2 (wte/wpe) + 2 (ln_f weight+bias) + 4 (ln_1/ln_2 × weight+bias) per layer
    @test length(sd_map) == 4 + cfg.num_hidden_layers * 4
end

@testset "load_state_dict! — GPT-2 slices fused QKV and transposes Conv1D" begin
    Random.seed!(0x9999)
    cfg = _tiny_gpt2_config()
    lm = GPT2ForCausalLM(cfg)
    sd = _gpt2_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    h = cfg.hidden_size

    # Embedding transpose: stored (vocab, hidden) → loaded (hidden, vocab).
    @test size(lm.model.embed_tokens.weight) == (h, cfg.vocab_size)
    @test lm.model.embed_tokens.weight == permutedims(sd["transformer.wte.weight"], (2, 1))
    @test lm.model.embed_positions.weight ==
        permutedims(sd["transformer.wpe.weight"], (2, 1))

    # LayerNorm biases flow.
    @test lm.model.norm.bias == sd["transformer.ln_f.bias"]
    @test lm.model.layers[1].input_layernorm.bias == sd["transformer.h.0.ln_1.bias"]

    # Fused QKV slicing: rebuild the expected slices from the source after transpose.
    qkv_src = permutedims(sd["transformer.h.0.attn.c_attn.weight"], (2, 1))
    @test lm.model.layers[1].self_attn.wq.weight == qkv_src[1:h, :]
    @test lm.model.layers[1].self_attn.wk.weight == qkv_src[(h + 1):(2 * h), :]
    @test lm.model.layers[1].self_attn.wv.weight == qkv_src[(2 * h + 1):(3 * h), :]

    qkv_b_src = sd["transformer.h.0.attn.c_attn.bias"]
    @test lm.model.layers[1].self_attn.wq.bias == qkv_b_src[1:h]
    @test lm.model.layers[1].self_attn.wk.bias == qkv_b_src[(h + 1):(2 * h)]
    @test lm.model.layers[1].self_attn.wv.bias == qkv_b_src[(2 * h + 1):(3 * h)]

    # Conv1D transpose on c_proj and MLP linears.
    @test lm.model.layers[1].self_attn.wo.weight ==
        permutedims(sd["transformer.h.0.attn.c_proj.weight"], (2, 1))
    @test lm.model.layers[1].self_attn.wo.bias == sd["transformer.h.0.attn.c_proj.bias"]
    @test lm.model.layers[1].mlp.c_fc.weight ==
        permutedims(sd["transformer.h.0.mlp.c_fc.weight"], (2, 1))
    @test lm.model.layers[1].mlp.c_fc.bias == sd["transformer.h.0.mlp.c_fc.bias"]
    @test lm.model.layers[1].mlp.c_proj.weight ==
        permutedims(sd["transformer.h.0.mlp.c_proj.weight"], (2, 1))
    @test lm.model.layers[1].mlp.c_proj.bias == sd["transformer.h.0.mlp.c_proj.bias"]

    # Tied embeddings: lm_head.weight is the transposed input embedding.
    @test lm.lm_head.weight == sd["transformer.wte.weight"]
end

@testset "load_state_dict! — GPT-2 errors on fused-QKV shape mismatch" begin
    cfg = _tiny_gpt2_config()
    lm = GPT2ForCausalLM(cfg)
    sd = _gpt2_synthetic_state_dict(cfg)
    sd["transformer.h.0.attn.c_attn.weight"] = randn(Float32, cfg.hidden_size, 7)   # wrong second dim
    @test_throws DimensionMismatch load_state_dict!(lm, sd)
end
