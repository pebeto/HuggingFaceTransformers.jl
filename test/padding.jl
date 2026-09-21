using Test
using Random
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Layers: GQA, RoPE, sdpa, flash_sdpa

# A padding mask marks which *keys* are real. It has to vary per sequence, so it
# carries a batch dimension that the shared causal/window masks do not.
@testset verbose = true "per-sequence attention masks" begin
    @testset "both attention paths agree on a 3-D mask" begin
        Random.seed!(0xA1)
        sq, skv, d, b = 4, 6, 8, 3
        q = randn(Float32, sq, d, b)
        k = randn(Float32, d, skv, b)
        v = randn(Float32, skv, d, b)
        drop = falses(sq, skv, b)
        drop[:, 5:6, 2] .= true                  # sequence 2 is padded

        @test isapprox(
            sdpa(q, k, v; scale=2.0f0, drop=drop),
            flash_sdpa(q, k, v; scale=2.0f0, drop=drop, block_size=3);
            rtol=1e-4,
        )
        # Masking one sequence must not touch the others.
        solo_drop = falses(sq, skv, 1)
        solo_drop[:, 5:6, 1] .= true
        batched = sdpa(q, k, v; scale=2.0f0, drop=drop)
        solo = sdpa(q[:, :, 2:2], k[:, :, 2:2], v[:, :, 2:2]; scale=2.0f0, drop=solo_drop)
        @test isapprox(batched[:, :, 2], solo[:, :, 1]; rtol=1e-5)
    end

    @testset "a 2-D mask still applies to every sequence" begin
        Random.seed!(0xA2)
        sq, skv, d, b = 4, 6, 8, 2
        q = randn(Float32, sq, d, b)
        k = randn(Float32, d, skv, b)
        v = randn(Float32, skv, d, b)
        drop = falses(sq, skv)
        drop[:, 5:6] .= true
        @test isapprox(
            sdpa(q, k, v; scale=2.0f0, drop=drop),
            flash_sdpa(q, k, v; scale=2.0f0, drop=drop, block_size=3);
            rtol=1e-4,
        )
    end

    @testset "bidirectional attention needs the mask" begin
        # Every position sees the padding, so masking is the only thing that makes
        # a padded batch match the unpadded sequence.
        Random.seed!(0xA3)
        gqa = GQA(16, 4, 4, 4, nothing; causal=false)
        x = randn(Float32, 16, 5, 1)
        mask = trues(5, 1)
        mask[4:5, 1] .= false

        masked = gqa(x; padding_mask=mask)
        shorter = gqa(x[:, 1:3, :])
        @test isapprox(masked[:, 1:3, 1], shorter[:, :, 1]; rtol=1e-5)
        @test !isapprox(gqa(x)[:, 1:3, 1], shorter[:, :, 1]; rtol=1e-3)
    end

    @testset "left padding needs the mask and honest positions" begin
        # Real tokens sit after the padding, so without a mask they attend to it.
        # The causal comparison also has to use each key's position rather than its
        # index, which differ as soon as the sequence does not start at 0.
        Random.seed!(0xA4)
        gqa = GQA(16, 4, 4, 4, RoPE(4; base=10000.0))
        x = randn(Float32, 16, 5, 1)
        mask = trues(5, 1)
        mask[1:2, 1] .= false
        positions = [0, 0, 0, 1, 2]              # pads park at 0; real tokens 0,1,2

        padded = gqa(x; padding_mask=mask, position_ids=positions)
        shorter = gqa(x[:, 3:5, :])
        @test isapprox(padded[:, 3:5, 1], shorter[:, :, 1]; rtol=1e-5)
        @test !isapprox(
            gqa(x; position_ids=positions)[:, 3:5, 1], shorter[:, :, 1]; rtol=1e-3
        )
    end

    @testset "an all-real mask changes nothing" begin
        Random.seed!(0xA5)
        gqa = GQA(16, 4, 2, 4, RoPE(4; base=10000.0))
        x = randn(Float32, 16, 5, 2)
        @test isapprox(gqa(x; padding_mask=trues(5, 2)), gqa(x); rtol=1e-6)
    end

    @testset "a mismatched mask is rejected" begin
        gqa = GQA(16, 4, 2, 4, RoPE(4; base=10000.0))
        x = randn(Float32, 16, 5, 2)
        @test_throws DimensionMismatch gqa(x; padding_mask=trues(4, 2))
        @test_throws DimensionMismatch gqa(x; padding_mask=trues(5, 3))
    end

    @testset "a padded batch matches per-sequence runs through a whole model" begin
        Random.seed!(0xA6)
        cfg = LlamaConfig(;
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=32, rope_theta=10000.0, tie_word_embeddings=false,
        )
        lm = LlamaForCausalLM(cfg)

        ids = [3 9; 7 2; 11 0; 5 0]              # second sequence is two tokens + padding
        mask = Bool[1 1; 1 1; 1 0; 1 0]
        batched = lm(ids; padding_mask=mask)

        @test isapprox(batched[:, :, 1], lm(reshape([3, 7, 11, 5], :, 1))[:, :, 1]; rtol=1e-5)
        @test isapprox(batched[:, 1:2, 2], lm(reshape([9, 2], :, 1))[:, :, 1]; rtol=1e-5)
    end
end
