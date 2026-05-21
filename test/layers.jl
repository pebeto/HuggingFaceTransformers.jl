using Test
using Allspark.Layers
using Flux
using NNlib
using Statistics

@testset "RMSNorm" begin
    dim = 8
    rms = RMSNorm(dim)
    @test length(rms.weight) == dim
    @test rms.eps == 1.0f-6

    # Test normalization mathematical correctness
    x = rand(Float32, dim, 3, 2)
    y = rms(x)
    @test size(y) == size(x)

    # RMS values of normalized outputs should be approximately 1
    rms_val = sqrt.(sum(y .^ 2, dims=1) ./ dim)
    @test all(isapprox.(rms_val, 1.0f0, atol=1e-3))

    # Test gradient flow and parameter updates
    grads = Flux.gradient(m -> sum(m(x)), rms)
    @test grads[1] !== nothing
    @test length(grads[1].weight) == dim
end

@testset "RoPE" begin
    dim = 8
    # 1. Unscaled RoPE
    rope = RoPE(dim; base=10000.0)
    @test length(rope.inv_freq) == dim ÷ 2

    # 2. Llama 3 scaled RoPE
    rope_scaled = RoPE(
        dim;
        base=500000.0,
        scaling_factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        old_context_len=8192,
    )
    @test length(rope_scaled.inv_freq) == dim ÷ 2
    @test rope.inv_freq != rope_scaled.inv_freq

    # Test rotation shapes
    # 2D
    x2d = rand(Float32, dim, 5)
    @test size(rope(x2d, 1:5)) == (dim, 5)

    # 3D
    x3d = rand(Float32, dim, 5, 2)
    @test size(rope(x3d, 1:5)) == (dim, 5, 2)

    # 4D
    x4d = rand(Float32, dim, 3, 5, 2)
    @test size(rope(x4d, 1:5)) == (dim, 3, 5, 2)

    # Mathematical check: rotation at position 0 is identity
    v = [1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0, 7.0f0, 8.0f0]
    @test isapprox(rope(v, [0.0f0]), v, atol=1e-5)

    # Analytical rotation check for position 1
    # For a 2D pair [x, y], rotated by theta: [x cos(theta) - y sin(theta), y cos(theta) + x sin(theta)]
    # With dim=2, base=10000, half_dim=1. inv_freq = [1.0]
    rope_2 = RoPE(2; base=1.0)
    v2 = [1.0f0, 0.0f0]
    # rotated by angle 1.0: [cos(1.0), sin(1.0)]
    v2_rot = rope_2(v2, [1.0f0])
    @test isapprox(v2_rot[1], cos(1.0f0), atol=1e-5)
    @test isapprox(v2_rot[2], sin(1.0f0), atol=1e-5)

    # Trainable parameters check (should be empty named tuple)
    @test Flux.Optimisers.trainable(rope) == (;)
end

@testset "Linear (no bias)" begin
    lin = Linear(4, 3)
    @test size(lin.weight) == (3, 4)

    x = rand(Float32, 4, 5, 2)
    y = lin(x)
    @test size(y) == (3, 5, 2)

    # Test gradient flow and parameter updates
    grads = Flux.gradient(m -> sum(m(x)), lin)
    @test grads[1] !== nothing
    @test size(grads[1].weight) == (3, 4)

    @test keys(Flux.Optimisers.trainable(lin)) == (:weight,)
end

@testset "SiLUGatedMLP" begin
    mlp = SiLUGatedMLP(4, 8)
    x = rand(Float32, 4, 5, 2)
    y = mlp(x)
    @test size(y) == (4, 5, 2)

    # Verify gradient flow
    grads = Flux.gradient(m -> sum(m(x)), mlp)
    @test grads[1] !== nothing
    @test keys(grads[1]) == (:gate_proj, :up_proj, :down_proj)

    # Verify trainable parameters
    @test keys(Flux.Optimisers.trainable(mlp)) == (:gate_proj, :up_proj, :down_proj)
end

@testset "GQA Attention" begin
    hidden_dim = 16
    num_heads_q = 4
    num_heads_k = 2
    head_dim = 8

    rope = RoPE(head_dim; base=10000.0)
    gqa = GQA(hidden_dim, num_heads_q, num_heads_k, head_dim, rope)

    x = rand(Float32, hidden_dim, 5, 2)
    y = gqa(x)
    @test size(y) == (hidden_dim, 5, 2)

    # 1. Verify trainable parameters
    @test keys(Flux.Optimisers.trainable(gqa)) == (:wq, :wk, :wv, :wo)

    # 2. Causal masking check (gradients of output at step t with respect to input at step > t should be 0)
    # Output of token 1: y[:, 1, 1]
    # Input of token 2: x[:, 2, 1]
    g = Flux.gradient(x -> sum(gqa(x)[:, 1, 1]), x)[1]
    @test all(g[:, 2:end, 1] .== 0)
    @test any(g[:, 1, 1] .!= 0)

    # 3. KV-cache verification
    # Preallocate cache (head_dim, n_kv_heads, max_seq, batch)
    cache = KVCache(head_dim, num_heads_k, 10, 2)
    @test size(cache.k) == (head_dim, num_heads_k, 10, 2)
    @test size(cache.v) == (head_dim, num_heads_k, 10, 2)

    # Passing a cache without a step must error
    @test_throws ArgumentError gqa(x; cache=cache)

    # First: Prefill 3 tokens (step 1)
    x1 = rand(Float32, hidden_dim, 3, 2)
    out1 = gqa(x1; cache=cache, step=1)
    @test size(out1) == (hidden_dim, 3, 2)

    # Verify cache has stored key/values for first 3 tokens
    # cache has batch size 2, n_kv_heads 2, max_seq 10, head_dim 8
    # Check that they are non-zero for first 3 indices and zero for subsequent indices
    @test any(cache.k[:, :, 1:3, :] .!= 0)
    @test all(cache.k[:, :, 4:10, :] .== 0)

    # Second: Decode 1 token (step 4)
    x2 = rand(Float32, hidden_dim, 1, 2)
    out2 = gqa(x2; cache=cache, step=4)
    @test size(out2) == (hidden_dim, 1, 2)
    @test any(cache.k[:, :, 4, :] .!= 0)
    @test all(cache.k[:, :, 5:10, :] .== 0)

    # Parity check: output from step-by-step decoding with cache must match prefill of the full 4-token sequence
    x_full = cat(x1, x2, dims=2)
    out_full = gqa(x_full)

    # Since GQA has randomly initialized weights, out_full[:, 4:4, :] should match out2 exactly (modulo floating point error)
    @test isapprox(out_full[:, 4:4, :], out2, atol=1e-4)
end
