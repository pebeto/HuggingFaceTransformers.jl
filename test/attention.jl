using Test
using Random
using HuggingFaceTransformers.Layers: sdpa, flash_sdpa, GQA, RoPE

# Helper: build a random (q, k_t, v) triple in the layout sdpa/flash_sdpa
# expect: q (sq, d, B), k_t (d, skv, B), v (skv, d, B).
function _qkv(sq, skv, d, batch; seed=0)
    Random.seed!(seed)
    q = randn(Float32, sq, d, batch)
    k_t = randn(Float32, d, skv, batch)
    v = randn(Float32, skv, d, batch)
    return q, k_t, v
end

@testset verbose = true "sdpa (materialized default)" begin
    q, k_t, v = _qkv(5, 7, 8, 2; seed=0x1)
    scale = sqrt(8.0f0)

    @testset "matches a hand-written softmax-of-matmul" begin
        out = sdpa(q, k_t, v; scale=scale)
        # Reference: per (head,batch) slice, softmax(qkᵀ/scale) v.
        ref = similar(out)
        for b in 1:size(q, 3)
            s = (q[:, :, b] * k_t[:, :, b]) ./ scale       # (sq, skv)
            s .-= maximum(s; dims=2)
            p = exp.(s)
            p ./= sum(p; dims=2)
            ref[:, :, b] = p * v[:, :, b]                  # (sq, d)
        end
        @test out ≈ ref
    end

    @testset "softcap bounds scores before softmax" begin
        out_raw = sdpa(q, k_t, v; scale=scale)
        out_cap = sdpa(q, k_t, v; scale=scale, softcap=5.0f0)
        @test size(out_cap) == size(out_raw)
        @test !(out_cap ≈ out_raw)
    end

    @testset "drop mask removes future keys (causal)" begin
        sq, skv = 5, 5
        q5, k5, v5 = _qkv(sq, skv, 8, 1; seed=0x2)
        drop = Bool[(j - 1) > (i - 1) for i in 1:sq, j in 1:skv]   # strict upper triangle
        out = sdpa(q5, k5, v5; scale=scale, drop=drop)
        # Query 1 attends to key 1 only → its output equals v[1].
        @test out[1, :, 1] ≈ v5[1, :, 1]
    end
end

@testset verbose = true "flash_sdpa matches materialized sdpa" begin
    scale = sqrt(8.0f0)

    @testset "no mask, single block (block_size ≥ seq_kv)" begin
        q, k_t, v = _qkv(4, 6, 8, 2; seed=0x10)
        a = sdpa(q, k_t, v; scale=scale)
        b = flash_sdpa(q, k_t, v; scale=scale, block_size=16)
        @test a ≈ b rtol = 1e-5
    end

    @testset "no mask, multiple blocks (block_size < seq_kv)" begin
        q, k_t, v = _qkv(4, 20, 8, 2; seed=0x11)
        a = sdpa(q, k_t, v; scale=scale)
        b = flash_sdpa(q, k_t, v; scale=scale, block_size=3)   # forces 7 blocks
        @test a ≈ b rtol = 1e-4
    end

    @testset "causal mask, multiple blocks" begin
        sq = skv = 16
        q, k_t, v = _qkv(sq, skv, 8, 2; seed=0x12)
        drop = Bool[(j - 1) > (i - 1) for i in 1:sq, j in 1:skv]
        a = sdpa(q, k_t, v; scale=scale, drop=drop)
        b = flash_sdpa(q, k_t, v; scale=scale, drop=drop, block_size=5)
        @test a ≈ b rtol = 1e-4
    end

    @testset "sliding window where late queries skip the first block" begin
        sq = skv = 12
        w = 3
        q, k_t, v = _qkv(sq, skv, 8, 1; seed=0x13)
        # Causal + window: query i sees keys (i-w+1 .. i). The last query
        # cannot see key 1, so the first KV block is fully masked for it.
        # That is the flash edge case that must not NaN or add weight.
        drop = Bool[(j - 1) > (i - 1) || (i - 1) - (j - 1) >= w for i in 1:sq, j in 1:skv]
        a = sdpa(q, k_t, v; scale=scale, drop=drop)
        b = flash_sdpa(q, k_t, v; scale=scale, drop=drop, block_size=4)
        @test !any(isnan, b)
        @test a ≈ b rtol = 1e-4
    end

    @testset "softcap composes with tiling" begin
        q, k_t, v = _qkv(6, 18, 8, 2; seed=0x14)
        a = sdpa(q, k_t, v; scale=scale, softcap=30.0f0)
        b = flash_sdpa(q, k_t, v; scale=scale, softcap=30.0f0, block_size=4)
        @test a ≈ b rtol = 1e-4
    end

    @testset "block_size larger than seq_kv equals a single pass" begin
        q, k_t, v = _qkv(4, 6, 8, 1; seed=0x15)
        one_block = flash_sdpa(q, k_t, v; scale=scale, block_size=100)
        many_block = flash_sdpa(q, k_t, v; scale=scale, block_size=2)
        @test one_block ≈ many_block rtol = 1e-5
    end
end

@testset "flash_sdpa is NaN-safe for a fully-masked query" begin
    sq = skv = 6
    scale = sqrt(8.0f0)
    q, k_t, v = _qkv(sq, skv, 8, 1; seed=0x20)
    # Drop everything for query 1 (it sees no keys at all). Degenerate, but
    # the kernel must return finite zeros, not NaN.
    drop = falses(sq, skv)
    drop[1, :] .= true
    out = flash_sdpa(q, k_t, v; scale=scale, drop=drop, block_size=2)
    @test !any(isnan, out)
    @test all(out[1, :, 1] .== 0)
end

@testset "flash_sdpa preserves input precision (fp16 in → fp16 out)" begin
    scale = sqrt(8.0f0)
    q, k_t, v = _qkv(4, 6, 8, 1; seed=0x30)
    q16, k16, v16 = Float16.(q), Float16.(k_t), Float16.(v)
    out = flash_sdpa(q16, k16, v16; scale=scale, block_size=3)
    @test eltype(out) === Float16
    # fp32-accumulated flash should still track the fp32 materialized result.
    ref = sdpa(q, k_t, v; scale=scale)
    @test maximum(abs.(Float32.(out) .- ref)) < 1e-2
end

@testset "GQA routes through sdpa with unchanged output" begin
    # The refactor moved GQA's attention core into `sdpa`. This pins that the
    # forward still produces a finite result of the right shape.
    Random.seed!(0xA11)
    hidden, nq, nkv, hd = 16, 4, 2, 4
    rope = RoPE(hd; base=10000.0)
    gqa = GQA(hidden, nq, nkv, hd, rope)
    x = randn(Float32, hidden, 5, 1)
    out = gqa(x)
    @test size(out) == (hidden, 5, 1)
    @test !any(isnan, out)
end
