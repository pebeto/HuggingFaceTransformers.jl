using Test
using Random
using Flux
using BFloat16s: BFloat16
using Allspark.Models
using Allspark.Models: build_caches, fp16, bf16, fp32, convert_eltype

# BFloat16 deadlocks in CPU LLVM codegen on Julia 1.12.6 (Float16 is fine).
# Gate the bf16 cases; opt in with ALLSPARK_TEST_BFLOAT16=1 where it works.
const BF16_OK = get(ENV, "ALLSPARK_TEST_BFLOAT16", "0") == "1"

# Tiny Llama so the conversion covers a real mix of layers and scalars.
function _tiny_llama_for_dtype()
    return LlamaConfig(;
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
    )
end

@testset verbose = true "dtype conversion" begin
    Random.seed!(0x1234)
    cfg = _tiny_llama_for_dtype()
    lm = LlamaForCausalLM(cfg)

    @testset "fp16 converts every Float32 leaf — arrays and scalars" begin
        lm16 = fp16(lm)
        @test eltype(lm16.lm_head.weight) === Float16
        @test eltype(lm16.model.embed_tokens.weight) === Float16
        @test eltype(lm16.model.norm.weight) === Float16
        # The RMSNorm `eps` scalar must flip too, else the norm promotes back
        # to fp32.
        @test typeof(lm16.model.norm.eps) === Float16
        # Layer 1 internals.
        layer = lm16.model.layers[1]
        @test eltype(layer.input_layernorm.weight) === Float16
        @test typeof(layer.input_layernorm.eps) === Float16
        @test eltype(layer.self_attn.wq.weight) === Float16
        @test eltype(layer.self_attn.rope.inv_freq) === Float16
        @test eltype(layer.mlp.gate_proj.weight) === Float16
    end

    if BF16_OK
        @testset "bf16 converts to BFloat16 with the same coverage" begin
            lm_bf = bf16(lm)
            @test eltype(lm_bf.lm_head.weight) === BFloat16
            @test eltype(lm_bf.model.embed_tokens.weight) === BFloat16
            @test typeof(lm_bf.model.norm.eps) === BFloat16
            @test eltype(lm_bf.model.layers[1].self_attn.rope.inv_freq) === BFloat16
        end
    end

    @testset "fp32 is the identity for a freshly-constructed model" begin
        lm32 = fp32(lm)
        @test eltype(lm32.lm_head.weight) === Float32
        # Values unchanged (no lossy round-trip on a same-precision conversion).
        @test lm32.lm_head.weight == lm.lm_head.weight
    end

    @testset "round-trip fp32 → fp16 → fp32 preserves shape" begin
        lm16 = fp16(lm)
        lm_back = fp32(lm16)
        @test size(lm_back.lm_head.weight) == size(lm.lm_head.weight)
        @test eltype(lm_back.lm_head.weight) === Float32
        # Values lose some precision but stay within Float16's nominal accuracy.
        @test isapprox(lm_back.lm_head.weight, lm.lm_head.weight; rtol=1e-2)
    end

    @testset "the original model is unchanged after conversion" begin
        _ = fp16(lm)
        @test eltype(lm.lm_head.weight) === Float32
    end

    @testset "convert_eltype dispatches on the type argument" begin
        @test eltype(convert_eltype(lm, Float16).lm_head.weight) === Float16
        @test eltype(convert_eltype(lm, Float64).lm_head.weight) === Float64
        if BF16_OK
            @test eltype(convert_eltype(lm, BFloat16).lm_head.weight) === BFloat16
        end
    end
end

@testset verbose = true "fp16/bf16 forward pass" begin
    Random.seed!(0x9999)
    cfg = _tiny_llama_for_dtype()
    lm = LlamaForCausalLM(cfg)

    # Capture the fp32 reference output for the same input so we can
    # bound the dtype-converted runs against it.
    ids = rand(0:(cfg.vocab_size - 1), 4, 1)
    out_fp32 = lm(ids)

    @testset "fp16 forward runs end-to-end" begin
        lm16 = fp16(lm)
        out16 = lm16(ids)
        @test size(out16) == (cfg.vocab_size, 4, 1)
        @test !any(isnan, out16)
        # Generous tolerance: the attention chain accumulates fp16 error.
        @test maximum(abs.(Float32.(out16) .- out_fp32)) < 1.0
    end

    if BF16_OK
        @testset "bf16 forward runs end-to-end" begin
            lm_bf = bf16(lm)
            out_bf = lm_bf(ids)
            @test size(out_bf) == (cfg.vocab_size, 4, 1)
            @test !any(isnan, out_bf)
            @test maximum(abs.(Float32.(out_bf) .- out_fp32)) < 1.0
        end
    end

    @testset "fp16 KV cache slot allocation matches model dtype" begin
        lm16 = fp16(lm)
        caches = build_caches(lm16, cfg.max_position_embeddings, 1; eltype=Float16)
        @test eltype(caches[1].k) === Float16
        # And the cached-decode forward path works through fp16 caches.
        out_prefill = lm16(ids[1:3, :]; caches=caches, step=1)
        out_decode = lm16(ids[4:4, :]; caches=caches, step=4)
        @test !any(isnan, out_prefill)
        @test !any(isnan, out_decode)
    end
end
