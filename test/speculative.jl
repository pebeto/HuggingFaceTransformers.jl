using Test
using Random
using Flux
using Allspark.Models
using Allspark.Generation

function _spec_tiny_lm(; vocab_size=16, seed=0x7777)
    cfg = LlamaConfig(;
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    Random.seed!(seed)
    return LlamaForCausalLM(cfg), cfg
end

@testset "greedy speculative ≡ greedy generate(target)" begin
    # The core invariant: regardless of the draft, greedy speculative decoding
    # must reproduce the target's greedy sequence byte-for-byte. A *different*
    # draft forces rejections (correction + cache rollback path); an *identical*
    # draft accepts every round (bonus + accept-all path).
    target, cfg = _spec_tiny_lm(; seed=0x1111)
    draft, _ = _spec_tiny_lm(; seed=0x2222)   # different weights, same vocab
    ids = [1, 2, 3, 4]

    for M in (1, 5, 7, 13)
        ref = generate(target, ids; max_new_tokens=M)
        for γ in (1, 2, 4, 8)
            spec = speculative_generate(target, draft, ids; max_new_tokens=M, n_draft=γ)
            @test spec == ref
            # An identical draft must agree too (different accept pattern).
            spec_same = speculative_generate(
                target, target, ids; max_new_tokens=M, n_draft=γ
            )
            @test spec_same == ref
        end
    end
end

@testset "speculative — shape & content" begin
    target, cfg = _spec_tiny_lm(; seed=0x3333)
    draft, _ = _spec_tiny_lm(; seed=0x4444)
    ids = [1, 2, 3]
    out = speculative_generate(target, draft, ids; max_new_tokens=5, n_draft=3)
    @test out isa Vector{Int}
    @test length(out) == 8
    @test out[1:3] == ids
    @test all(0 .<= out .< cfg.vocab_size)
end

@testset "speculative — max_new_tokens cap is exact" begin
    # γ+1 tokens can land per round; the cap must still be honoured exactly,
    # including when max_new_tokens is not a multiple of (γ+1).
    target, _ = _spec_tiny_lm(; seed=0x5555)
    draft, _ = _spec_tiny_lm(; seed=0x6666)
    ids = [2, 4, 6]
    for M in (1, 2, 3, 4, 10)
        out = speculative_generate(target, draft, ids; max_new_tokens=M, n_draft=4)
        @test length(out) == length(ids) + M
    end
end

@testset "speculative — max_new_tokens=0 short-circuits" begin
    target, _ = _spec_tiny_lm()
    draft, _ = _spec_tiny_lm(; seed=0x9999)
    @test speculative_generate(target, draft, [1, 2, 3]; max_new_tokens=0) == [1, 2, 3]
end

@testset "speculative — EOS terminates" begin
    target, _ = _spec_tiny_lm(; seed=0x1111)
    draft, _ = _spec_tiny_lm(; seed=0x2222)
    ids = [1, 2, 3, 4]
    greedy = generate(target, ids; max_new_tokens=6)
    eos = greedy[length(ids) + 1]   # the first token the target would emit
    out = speculative_generate(target, draft, ids; max_new_tokens=6, n_draft=4, eos_token_id=eos)
    @test out[end] == eos
    @test length(out) == length(ids) + 1
end

@testset "speculative — sampling is reproducible with a fixed seed" begin
    target, _ = _spec_tiny_lm(; seed=0x7777)
    draft, _ = _spec_tiny_lm(; seed=0x8888)
    ids = [1, 2, 3]
    out1 = speculative_generate(
        target, draft, ids;
        max_new_tokens=8, n_draft=4, do_sample=true, temperature=0.9,
        rng=MersenneTwister(2024),
    )
    out2 = speculative_generate(
        target, draft, ids;
        max_new_tokens=8, n_draft=4, do_sample=true, temperature=0.9,
        rng=MersenneTwister(2024),
    )
    @test out1 == out2
    @test out1[1:3] == ids
end

@testset "speculative — vocab mismatch is rejected" begin
    target, _ = _spec_tiny_lm(; vocab_size=16, seed=0x1010)
    draft, _ = _spec_tiny_lm(; vocab_size=20, seed=0x2020)
    @test_throws DimensionMismatch speculative_generate(
        target, draft, [1, 2, 3]; max_new_tokens=4
    )
end

@testset "speculative — argument validation" begin
    target, _ = _spec_tiny_lm()
    draft, _ = _spec_tiny_lm(; seed=0x3030)
    @test_throws ArgumentError speculative_generate(target, draft, Int[]; max_new_tokens=5)
    @test_throws ArgumentError speculative_generate(
        target, draft, [1]; max_new_tokens=-1
    )
    @test_throws ArgumentError speculative_generate(
        target, draft, [1]; max_new_tokens=5, n_draft=0
    )
end
