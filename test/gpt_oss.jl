using Test
using Random
using JSON3
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models:
    GptOssConfig, GptOssForCausalLM, GptOssMLP, GptOssExperts, MXFP4Weights, decode_mxfp4!,
    gpt_oss_state_dict_map, load_pretrained, build_caches, _resolve, _save_safetensors,
    _clamped_swiglu
using HuggingFaceTransformers.Layers:
    Linear, RoPE, YarnRopeScaling, sdpa, flash_sdpa, _softmax_with_sinks

const _FP4 = Float32[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

# A random expert stack MXFP4 represents exactly: FP4 values times one power of
# two per 32-wide block. Returns the packed form and the same weights dense,
# `(n_out, n_in, n_experts)`.
function _random_mxfp4(rng, n_out, n_in, n_experts)
    n_blocks = n_in ÷ 32
    codes = rand(rng, 0:15, 32, n_blocks, n_out, n_experts)
    exponents = rand(rng, 118:124, n_blocks, n_out, n_experts)   # scales 2^-9 to 2^-3
    blocks = Array{UInt8}(undef, 16, n_blocks, n_out, n_experts)
    for b in 1:16
        @. blocks[b, :, :, :] = UInt8(codes[2b - 1, :, :, :] | (codes[2b, :, :, :] << 4))
    end
    dense = Array{Float32}(undef, n_out, n_in, n_experts)
    for e in 1:n_experts, r in 1:n_out, g in 1:n_blocks, j in 1:32
        dense[r, 32(g - 1) + j, e] = ldexp(_FP4[codes[j, g, r, e] + 1], exponents[g, r, e] - 127)
    end
    return MXFP4Weights(blocks, UInt8.(exponents)), dense
end

function _tiny_gpt_oss_config(; num_hidden_layers=2)
    return GptOssConfig(;
        vocab_size=64, hidden_size=64, intermediate_size=32, num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64, sliding_window=4,
        rope_scaling=YarnRopeScaling(; factor=4.0, original_max_position_embeddings=16, truncate=false),
    )
end

# A random model whose experts MXFP4 holds exactly, with its packed experts.
function _tiny_gpt_oss(rng)
    cfg = _tiny_gpt_oss_config()
    lm = GptOssForCausalLM(cfg)
    packed = map(lm.model.layers) do layer
        ex = layer.mlp.experts
        gu, gu_dense = _random_mxfp4(rng, 2cfg.intermediate_size, cfg.hidden_size, cfg.num_local_experts)
        dn, dn_dense = _random_mxfp4(rng, cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts)
        ex.gate_up .= gu_dense
        ex.down .= dn_dense
        ex.gate_up_bias .= randn(rng, Float32, size(ex.gate_up_bias)) .* 0.1f0
        ex.down_bias .= randn(rng, Float32, size(ex.down_bias)) .* 0.1f0
        layer.self_attn.sinks .= randn(rng, Float32, length(layer.self_attn.sinks))
        (; gu, dn)
    end
    return cfg, lm, packed
end

# HF-named tensors for every non-expert parameter, plus each layer's experts as
# HF stores them: packed MXFP4 (`(E, out, blocks, 16)` / `(E, out, blocks)`) or
# dense (`(E, in, out)`).
function _checkpoint(lm, packed; mxfp4::Bool)
    cfg = lm.config
    tensors = Dict{String,AbstractArray}()
    for (key, (path, transform)) in gpt_oss_state_dict_map(cfg)
        a = _resolve(lm, path)
        tensors[key] = transform === :transpose ? permutedims(a, (2, 1)) : copy(a)
    end
    for (i, layer) in enumerate(lm.model.layers)
        prefix = "model.layers.$(i - 1).mlp.experts."
        ex = layer.mlp.experts
        for (name, dense, w) in (("gate_up_proj", ex.gate_up, packed[i].gu), ("down_proj", ex.down, packed[i].dn))
            if mxfp4
                tensors[prefix * name * "_blocks"] = permutedims(w.blocks, (4, 3, 2, 1))
                tensors[prefix * name * "_scales"] = permutedims(w.scales, (3, 2, 1))
            else
                tensors[prefix * name] = permutedims(dense, (3, 2, 1))
            end
        end
    end
    return tensors
end

@testset verbose = true "MXFP4" begin
    @testset "nibbles decode low first, scaled by their block" begin
        blocks = zeros(UInt8, 16, 1, 1, 1)
        blocks[1] = 0x21         # low nibble 1 → 0.5, high nibble 2 → 1.0
        blocks[2] = 0xf8         # low nibble 8 → -0.0, high nibble 15 → -6.0
        w = MXFP4Weights(blocks, fill(UInt8(128), 1, 1, 1))   # scale 2^(128 - 127)
        buffer = decode_mxfp4!(Matrix{Float32}(undef, 32, 1), w, 1)
        @test buffer[1:4, 1] == Float32[1.0, 2.0, -0.0, -12.0]
        @test all(iszero, buffer[5:end, 1])
    end

    @testset "decoding reproduces the dense weights" begin
        w, dense = _random_mxfp4(Xoshiro(1), 6, 64, 3)
        buffer = Matrix{Float32}(undef, 64, 6)
        @test all(transpose(decode_mxfp4!(buffer, w, e)) == dense[:, :, e] for e in 1:3)
        @test_throws DimensionMismatch decode_mxfp4!(Matrix{Float32}(undef, 6, 64), w, 1)
        @test_throws DimensionMismatch MXFP4Weights(zeros(UInt8, 8, 1, 1, 1), zeros(UInt8, 1, 1, 1))
    end
end

@testset verbose = true "sink attention" begin
    rng = Xoshiro(2)
    scores = randn(rng, Float32, 3, 5, 2)
    sinks = Float32[0.7, -1.2]
    expected = similar(scores)
    for b in 1:2, i in 1:3
        e = exp.(scores[i, :, b])
        expected[i, :, b] = e ./ (sum(e) + exp(sinks[b]))
    end
    @test _softmax_with_sinks(scores, sinks) ≈ expected

    q, k_t, v = randn(rng, Float32, 7, 8, 6), randn(rng, Float32, 8, 9, 6), randn(rng, Float32, 9, 8, 6)
    s = randn(rng, Float32, 6)
    with_sinks = sdpa(q, k_t, v; scale=sqrt(8.0f0), sinks=s)
    # The tiled path seeds its running softmax with the sink: same result.
    @test flash_sdpa(q, k_t, v; scale=sqrt(8.0f0), sinks=s, block_size=4) ≈ with_sinks
    # A sink of -Inf takes no mass, which is plain attention.
    @test sdpa(q, k_t, v; scale=sqrt(8.0f0), sinks=fill(-Inf32, 6)) ≈ sdpa(q, k_t, v; scale=sqrt(8.0f0))
    @test !(with_sinks ≈ sdpa(q, k_t, v; scale=sqrt(8.0f0)))
end

@testset "YaRN matches HF's frequencies" begin
    # gpt-oss-20b's RoPE, values read from transformers' `rotary_emb.inv_freq`.
    rope = RoPE(64; base=150_000, yarn=YarnRopeScaling(;
        factor=32.0, original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        truncate=false,
    ))
    @test rope.attention_scaling ≈ 1.3465735902799727
    hf = [1.0, 0.15532298386096954, 0.019334999844431877, 0.0004564839182421565,
        1.818833698052913e-5, 3.023511396804679e-7]
    @test rope.inv_freq[[1, 6, 11, 17, 21, 32]] ≈ hf rtol = 1.0e-6
    @test RoPE(64; base=10_000).attention_scaling == 1
end

@testset "clamped SwiGLU" begin
    # Rows alternate gate, up. The gate is capped at the limit, up clamped both ways.
    gate_up = reshape(Float32[10, -9, -1, 0.5], 4, 1)
    act = _clamped_swiglu(gate_up, 1.702f0, 7.0f0)
    swish(g) = g / (1 + exp(-1.702f0 * g))
    @test act ≈ [(-7 + 1) * swish(7.0f0); (0.5f0 + 1) * swish(-1.0f0)]
end

@testset "router weights are a softmax over the chosen logits" begin
    # Every expert outputs its own index: zero weights, bias = e. With the router
    # reading only its bias, the mixture is fixed and checkable by hand.
    hidden, inter, n_experts = 32, 32, 4
    experts = GptOssExperts(
        zeros(Float32, 2inter, hidden, n_experts), zeros(Float32, 2inter, n_experts),
        zeros(Float32, hidden, inter, n_experts), repeat(Float32.(1:n_experts)', hidden, 1),
        1.702f0, 7.0f0,
    )
    router = Linear(zeros(Float32, n_experts, hidden), Float32[0.1, 2.0, -1.0, 1.5])
    mlp = GptOssMLP(router, experts, 2)
    out = mlp(randn(Float32, hidden, 3, 1))
    p = exp.(Float32[2.0, 1.5]) ./ sum(exp.(Float32[2.0, 1.5]))   # experts 2 and 4
    @test all(out .≈ p[1] * 2 + p[2] * 4)
end

@testset verbose = true "GptOssForCausalLM" begin
    cfg, lm, packed = _tiny_gpt_oss(Xoshiro(3))
    ids = reshape(rand(Xoshiro(4), 0:63, 10), :, 1)
    logits = lm(ids)

    @test size(logits) == (64, 10, 1)
    @test all(isfinite, logits)
    @test [layer.self_attn.window_size for layer in lm.model.layers] == [4, nothing]
    @test lm.model.layers[1].self_attn.rope.attention_scaling > 1

    @testset "KV-cache decode matches the full pass past the window" begin
        caches = build_caches(lm, cfg.max_position_embeddings, 1)
        lm(ids[1:7, :]; caches=caches, step=1)
        steps = [lm(ids[t:t, :]; caches=caches, step=t) for t in 8:10]
        @test all(isapprox(steps[j], logits[:, 7 + j:7 + j, :]; atol=1.0e-4) for j in 1:3)
    end

    @testset "packed and dense experts give the same model" begin
        packed_lm = GptOssForCausalLM(cfg, _checkpoint(lm, packed; mxfp4=true))
        @test packed_lm.model.layers[1].mlp.experts.gate_up isa MXFP4Weights
        @test packed_lm(ids) ≈ logits

        dense_lm = GptOssForCausalLM(cfg, _checkpoint(lm, packed; mxfp4=false))
        @test dense_lm.model.layers[1].mlp.experts.gate_up isa Array{Float32,3}
        @test dense_lm(ids) ≈ logits

        # Loading packed experts into an existing dense model decodes them.
        @test load_state_dict!(GptOssForCausalLM(cfg), _checkpoint(lm, packed; mxfp4=true))(ids) ≈ logits
    end

    @testset "load_pretrained keeps a checkpoint's MXFP4 experts packed" begin
        mktempdir() do dir
            _save_safetensors(joinpath(dir, "model.safetensors"), _checkpoint(lm, packed; mxfp4=true))
            write(joinpath(dir, "config.json"), JSON3.write(Dict(
                "architectures" => ["GptOssForCausalLM"], "model_type" => "gpt_oss",
                "vocab_size" => 64, "hidden_size" => 64, "intermediate_size" => 32,
                "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
                "head_dim" => 16, "num_local_experts" => 4, "num_experts_per_tok" => 2,
                "max_position_embeddings" => 64, "sliding_window" => 4, "rms_norm_eps" => 1.0e-5,
                "layer_types" => ["sliding_attention", "full_attention"],
                "rope_theta" => 150_000,
                "rope_scaling" => Dict("rope_type" => "yarn", "factor" => 4.0,
                    "original_max_position_embeddings" => 16, "beta_fast" => 32.0,
                    "beta_slow" => 1.0, "truncate" => false),
            )))
            loaded = load_pretrained(dir)
            @test loaded isa GptOssForCausalLM
            # Zero-copy: the experts are views over the memory-mapped file.
            @test !(loaded.model.layers[1].mlp.experts.gate_up.blocks isa Array)
            @test loaded(ids) ≈ logits
        end
    end
end

@testset "config_from_json(GptOssConfig)" begin
    base = Dict(
        "vocab_size" => 64, "hidden_size" => 64, "intermediate_size" => 32,
        "num_hidden_layers" => 4, "num_attention_heads" => 4, "num_key_value_heads" => 2,
        "head_dim" => 16, "num_local_experts" => 4, "max_position_embeddings" => 64,
    )
    parse(extra) = config_from_json(GptOssConfig, JSON3.read(JSON3.write(merge(base, extra))))

    plain = parse(Dict())
    @test isnothing(plain.rope_scaling)
    @test plain.layer_types == [:sliding_attention, :full_attention, :sliding_attention, :full_attention]
    @test plain.num_experts_per_tok == 4 && plain.attention_bias

    yarn = Dict("rope_type" => "yarn", "factor" => 32.0, "original_max_position_embeddings" => 4096,
        "truncate" => false, "rope_theta" => 150_000)
    cfg = parse(Dict("rope_parameters" => yarn))
    @test cfg.rope_theta == 150_000
    @test cfg.rope_scaling.factor == 32 && !cfg.rope_scaling.truncate
    @test parse(Dict("rope_scaling" => yarn, "rope_theta" => 10.0)).rope_theta == 10
    @test_throws ArgumentError parse(Dict("rope_scaling" => Dict("rope_type" => "dynamic", "factor" => 2.0)))
    @test_throws ArgumentError GptOssForCausalLM(parse(Dict("layer_types" => fill("chunked_attention", 4))))
end
