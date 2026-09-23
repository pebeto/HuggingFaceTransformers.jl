using Test
using Random
using Flux
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Generation
using HuggingFaceTransformers.Tokenizers
using HuggingFaceTransformers.Tokenizers:
    BPEModel, ByteLevelPreTokenizer, ByteLevelDecoder, BYTES_TO_UNICODE, AddedToken

function _tiny_lm(; vocab_size=16, seed=0x7777)
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

@testset "generate — shape & content" begin
    lm, cfg = _tiny_lm()
    ids = [1, 2, 3]
    out = generate(lm, ids; max_new_tokens=5)
    @test out isa Vector{Int}
    @test length(out) == 8
    @test out[1:3] == ids
    @test all(0 .<= out .< cfg.vocab_size)
end

@testset "generate — max_new_tokens=0 short-circuits" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    @test generate(lm, ids; max_new_tokens=0) == ids
end

@testset "generate — greedy is deterministic across calls" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    @test generate(lm, ids; max_new_tokens=6) == generate(lm, ids; max_new_tokens=6)
end

@testset "generate — sampling reproducible with same RNG seed" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    out1 = generate(
        lm, ids; max_new_tokens=6, do_sample=true, temperature=0.8, rng=MersenneTwister(123)
    )
    out2 = generate(
        lm, ids; max_new_tokens=6, do_sample=true, temperature=0.8, rng=MersenneTwister(123)
    )
    @test out1 == out2
end

@testset "generate — different RNG seeds diverge under high entropy" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    out1 = generate(
        lm, ids; max_new_tokens=12, do_sample=true, temperature=2.0, rng=MersenneTwister(1)
    )
    out2 = generate(
        lm, ids; max_new_tokens=12, do_sample=true, temperature=2.0, rng=MersenneTwister(2)
    )
    @test out1 != out2
end

@testset "generate — top_k=1 ≡ greedy" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    greedy = generate(lm, ids; max_new_tokens=6)
    topk1 = generate(
        lm, ids; max_new_tokens=6, do_sample=true, top_k=1, rng=MersenneTwister(42)
    )
    @test greedy == topk1
end

@testset "generate — EOS terminates" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    greedy = generate(lm, ids; max_new_tokens=6)
    eos = greedy[length(ids) + 1]   # whatever the model would emit first
    out = generate(lm, ids; max_new_tokens=6, eos_token_id=eos)
    @test out[end] == eos
    @test length(out) == length(ids) + 1
end

@testset "generate — EOS accepts a vector of IDs" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    greedy = generate(lm, ids; max_new_tokens=6)
    eos = greedy[length(ids) + 1]
    out = generate(lm, ids; max_new_tokens=6, eos_token_id=[eos, eos + 1])
    @test out[end] in (eos, eos + 1)
end

@testset "generate — repetition penalty perturbs greedy output" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3]
    baseline = generate(lm, ids; max_new_tokens=6)
    penalized = generate(lm, ids; max_new_tokens=6, repetition_penalty=100.0)
    # With a brutally large penalty, tokens already in `ids` should be heavily
    # suppressed — the output must differ from the unpenalized greedy run.
    @test baseline != penalized
end

@testset "generate — argument validation" begin
    lm, _ = _tiny_lm()
    @test_throws ArgumentError generate(lm, Int[]; max_new_tokens=5)
    @test_throws ArgumentError generate(lm, [1]; max_new_tokens=-1)
    @test_throws ArgumentError generate(
        lm, [1]; max_new_tokens=5, do_sample=true, temperature=0.0
    )
    @test_throws ArgumentError generate(lm, [1]; max_new_tokens=5, do_sample=true, top_k=0)
    @test_throws ArgumentError generate(
        lm, [1]; max_new_tokens=5, do_sample=true, top_p=0.0
    )
    @test_throws ArgumentError generate(lm, [1]; max_new_tokens=5, repetition_penalty=0.0)
end

@testset "generate(lm, tokenizer, prompt)" begin
    # Build a minimal byte-level tokenizer over the first 256 IDs.
    vocab = Dict{String,Int}()
    for b in 0x00:0xff
        vocab[string(BYTES_TO_UNICODE[b])] = Int(b)
    end
    tk = Tokenizer(
        BPEModel(vocab, Tuple{String,String}[]),
        HuggingFaceTransformers.Tokenizers.IdentityNormalizer(),
        ByteLevelPreTokenizer(false, true),
        ByteLevelDecoder(),
        AddedToken[],
        Dict{String,Int}(),
        Dict{Int,String}(),
    )

    lm, _ = _tiny_lm(; vocab_size=256, seed=0x8888)

    out = generate(lm, tk, "hello"; max_new_tokens=3)
    @test out isa String
    # Prompt is preserved at the front of the decoded output.
    @test startswith(out, "hello")
end

@testset verbose = true "batched generate" begin
    # Prompts of differing length are left-padded so one shared decode step can
    # advance every row. Positions are the padded indices, which is exact for
    # rotary models because RoPE depends on query-key position differences.
    function _batch_lm(; seed=0xC2)
        cfg = LlamaConfig(;
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=64, rope_theta=10000.0, tie_word_embeddings=false,
        )
        Random.seed!(seed)
        return LlamaForCausalLM(cfg)
    end

    PROMPTS = [[3, 7, 11, 5], [9, 2], [4]]

    @testset "greedy matches running each prompt alone" begin
        # The load-bearing test: padding, masking, position offsets and cache
        # indexing all have to be right for these to agree token for token.
        lm = _batch_lm()
        batched = generate(lm, PROMPTS; max_new_tokens=6)
        for (b, prompt) in enumerate(PROMPTS)
            @test batched[b] == generate(lm, prompt; max_new_tokens=6)
        end
    end

    @testset "each row stops at its own EOS" begin
        lm = _batch_lm()
        free = generate(lm, PROMPTS; max_new_tokens=6)
        # Whatever the shortest prompt produces first becomes its EOS, so that row
        # stops immediately while the others run on.
        eos = free[3][length(PROMPTS[3]) + 1]
        stopped = generate(lm, PROMPTS; max_new_tokens=6, eos_token_id=eos)

        @test stopped[3] == [PROMPTS[3]; eos]            # EOS is emitted, then stop
        @test length(stopped[1]) > length(PROMPTS[1]) + 1
        # Rows that never hit EOS are unaffected by another row finishing.
        @test stopped[1] == free[1]
        @test stopped[2] == free[2]
    end

    @testset "prompts are returned intact" begin
        lm = _batch_lm()
        out = generate(lm, PROMPTS; max_new_tokens=4)
        for (b, prompt) in enumerate(PROMPTS)
            @test out[b][1:length(prompt)] == prompt
            @test length(out[b]) <= length(prompt) + 4
        end
    end

    @testset "max_new_tokens=0 returns the prompts" begin
        lm = _batch_lm()
        @test generate(lm, PROMPTS; max_new_tokens=0) == PROMPTS
    end

    @testset "argument validation" begin
        lm = _batch_lm()
        @test_throws ArgumentError generate(lm, Vector{Int}[]; max_new_tokens=2)
        @test_throws ArgumentError generate(lm, [[1, 2], Int[]]; max_new_tokens=2)
        @test_throws ArgumentError generate(lm, PROMPTS; max_new_tokens=-1)
    end

    @testset "Gemma batches, sliding window included" begin
        # Gemma has its own trunk, so it needs the mask threaded separately. The
        # window of 4 engages on the 6-token prompt, and both softcaps are live.
        cfg = GemmaConfig(;
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=64, sliding_window=4, attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0, query_pre_attn_scalar=4,
        )
        Random.seed!(0xD2)
        lm = GemmaForCausalLM(cfg)
        prompts = [[3, 7, 11, 5, 9, 2], [9, 2], [4]]
        batched = generate(lm, prompts; max_new_tokens=6)
        for (b, prompt) in enumerate(prompts)
            @test batched[b] == generate(lm, prompt; max_new_tokens=6)
        end
    end

    @testset "GPT-2 batches with per-row learned positions" begin
        # GPT-2's positions are learned embeddings, so an offset changes the vector
        # rather than cancelling as it does under RoPE. Each left-padded row has to
        # count its own real tokens.
        cfg = GPT2Config(;
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=64,
        )
        Random.seed!(0xD1)
        lm = GPT2ForCausalLM(cfg)
        batched = generate(lm, PROMPTS; max_new_tokens=6)
        for (b, prompt) in enumerate(PROMPTS)
            @test batched[b] == generate(lm, prompt; max_new_tokens=6)
        end

        # Guard against a vacuous pass: padding must genuinely perturb GPT-2, so
        # treating the pads as real tokens has to give a different answer.
        ids = [0 3; 0 7; 9 11; 2 5]                       # row 1 is two pads + [9, 2]
        real = Bool[0 1; 0 1; 1 1; 1 1]
        solo = lm(reshape([9, 2], :, 1))[:, :, 1]
        @test isapprox(lm(ids; padding_mask=real)[:, 3:4, 1], solo; rtol=1e-5)
        @test !isapprox(lm(ids; padding_mask=trues(4, 2))[:, 3:4, 1], solo; rtol=1e-3)
    end

    @testset "a uniform batch needs no padding at all" begin
        lm = _batch_lm()
        same = [[3, 7, 11], [9, 2, 5]]
        batched = generate(lm, same; max_new_tokens=5)
        for (b, prompt) in enumerate(same)
            @test batched[b] == generate(lm, prompt; max_new_tokens=5)
        end
    end
end
