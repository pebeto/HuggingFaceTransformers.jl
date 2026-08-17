# GPU attention verification. Not part of the default suite: it needs a real
# device, because the point is that loading a backend triggers the matching
# HuggingFaceTransformers extension and routes `sdpa` on device arrays to
# `flash_sdpa`. This checks that path against the CPU materialized one.
#
#   HFT_GPU_BACKEND=cuda julia --project=<env-with-CUDA> test/gpu_attention.jl
using Test
using Random
using Flux
using NNlib
using HuggingFaceTransformers.Layers: sdpa, flash_sdpa

include("gpu_backend.jl")

GPU_BACKEND == "jlarrays" && error(
    "gpu_attention.jl needs real hardware (the extension does not load for the " *
    "reference backend); set HFT_GPU_BACKEND=cuda|amdgpu|metal",
)

function _qkv(sq, skv, d, batch; seed=0)
    Random.seed!(seed)
    return (
        randn(Float32, sq, d, batch),
        randn(Float32, d, skv, batch),
        randn(Float32, skv, d, batch),
    )
end

@testset verbose = true "GPU attention ($(BACKEND_NAME))" begin
    scale = sqrt(8.0f0)

    @testset "device sdpa matches CPU materialized (no mask)" begin
        q, k_t, v = _qkv(4, 20, 8, 2; seed=0x10)
        cpu = sdpa(q, k_t, v; scale=scale)
        gpu = sdpa(dev(q), dev(k_t), dev(v); scale=scale)
        @test dev_approx(gpu, cpu)
    end

    @testset "device sdpa matches CPU (causal mask)" begin
        sq = skv = 16
        q, k_t, v = _qkv(sq, skv, 8, 2; seed=0x12)
        drop = Bool[(j - 1) > (i - 1) for i in 1:sq, j in 1:skv]
        cpu = sdpa(q, k_t, v; scale=scale, drop=drop)
        gpu = sdpa(dev(q), dev(k_t), dev(v); scale=scale, drop=drop)
        @test dev_approx(gpu, cpu)
    end

    @testset "device sdpa matches CPU (softcap + sliding window)" begin
        sq = skv = 12
        w = 4
        q, k_t, v = _qkv(sq, skv, 8, 1; seed=0x13)
        drop = Bool[(j - 1) > (i - 1) || (i - 1) - (j - 1) >= w for i in 1:sq, j in 1:skv]
        cpu = sdpa(q, k_t, v; scale=scale, softcap=30.0f0, drop=drop)
        gpu = sdpa(
            dev(q),
            dev(k_t),
            dev(v);
            scale=scale,
            softcap=30.0f0,
            drop=drop,
        )
        @test !any(isnan, Array(gpu))
        @test dev_approx(gpu, cpu)
    end

    @testset "device sdpa actually dispatched to flash_sdpa" begin
        # If the extension didn't load, `sdpa` would fall through to the
        # materialized CPU method and error on device arrays inside
        # `batched_mul`/`softmax`. Reaching here at all is the check.
        q, k_t, v = _qkv(3, 5, 8, 1; seed=0x20)
        gpu = sdpa(dev(q), dev(k_t), dev(v); scale=scale)
        @test gpu isa typeof(dev(q))
    end
end
