using Test
using Random
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models: Linear, QuantizedInt8Matrix, quantize_int8

function _tiny_llama_for_quant()
    return LlamaConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
    )
end

@testset verbose = true "QuantizedInt8Matrix" begin
    Random.seed!(0x1234)
    W = randn(Float32, 6, 10)
    q = QuantizedInt8Matrix(W)

    @testset "shape, eltype, storage" begin
        @test size(q) == size(W)
        @test eltype(q) === Float32
        @test q.q isa Matrix{Int8}
        @test length(q.scale) == size(W, 1)
    end

    @testset "getindex dequantizes within one quantization step" begin
        for i in axes(W, 1), j in axes(W, 2)
            @test abs(q[i, j] - W[i, j]) <= q.scale[i]
        end
    end

    @testset "matmul matches the dequantized weight" begin
        x = randn(Float32, 10, 4)
        deq = [q[i, j] for i in axes(q, 1), j in axes(q, 2)]
        @test q * x ≈ deq * x
    end

    @testset "all-zero row gets a safe scale" begin
        Wz = zeros(Float32, 3, 5)
        qz = QuantizedInt8Matrix(Wz)
        @test all(qz.scale .== 1.0f0)
        @test all(qz.q .== 0)
    end
end

@testset verbose = true "quantize_int8(model)" begin
    Random.seed!(0x9999)
    cfg = _tiny_llama_for_quant()
    lm = LlamaForCausalLM(cfg)
    lmq = quantize_int8(lm)

    @testset "Linear weights become QuantizedInt8Matrix" begin
        @test lmq.lm_head.weight isa QuantizedInt8Matrix
        layer = lmq.model.layers[1]
        @test layer.self_attn.wq.weight isa QuantizedInt8Matrix
        @test layer.self_attn.wo.weight isa QuantizedInt8Matrix
        @test layer.mlp.gate_proj.weight isa QuantizedInt8Matrix
        @test layer.mlp.down_proj.weight isa QuantizedInt8Matrix
    end

    @testset "embeddings and norms stay Float32" begin
        @test lmq.model.embed_tokens.weight isa Matrix{Float32}
        @test lmq.model.norm.weight isa Vector{Float32}
        @test lmq.model.layers[1].input_layernorm.weight isa Vector{Float32}
    end

    @testset "the original model is unquantized" begin
        @test lm.lm_head.weight isa Matrix{Float32}
    end

    @testset "forward runs and tracks the fp32 logits" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        out_fp32 = lm(ids)
        out_q = lmq(ids)
        @test size(out_q) == (cfg.vocab_size, 4, 1)
        @test !any(isnan, out_q)
        @test maximum(abs.(out_q .- out_fp32)) < 1.0
    end

    @testset "KV-cache decode works through quantized weights" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        caches = build_caches(lmq, cfg.max_position_embeddings, 1)
        out_prefill = lmq(ids[1:3, :]; caches=caches, step=1)
        out_decode = lmq(ids[4:4, :]; caches=caches, step=4)
        @test !any(isnan, out_prefill)
        @test !any(isnan, out_decode)
    end
end

@testset "quantize_int8 also quantizes a Qwen model (biased QKV)" begin
    Random.seed!(0xBEEF)
    cfg = QwenConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
    )
    lmq = quantize_int8(QwenForCausalLM(cfg))
    layer = lmq.model.layers[1]
    # Weight quantized; the Q/K/V bias vectors stay Float32.
    @test layer.self_attn.wq.weight isa QuantizedInt8Matrix
    @test layer.self_attn.wq.bias isa Vector{Float32}
    ids = rand(0:(cfg.vocab_size - 1), 3, 1)
    @test !any(isnan, lmq(ids))
end
