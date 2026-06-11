using Test
using Random
using Flux
using Allspark.Models
using Allspark.Generation
using Allspark.Tokenizers
using Allspark.Tokenizers:
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
        Allspark.Tokenizers.IdentityNormalizer(),
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
