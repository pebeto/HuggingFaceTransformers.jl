# GPU attention verification. Not part of the default suite. Run it by hand
# on a machine with a functional CUDA / AMDGPU / Metal device:
#
#   julia --project=. test/gpu_attention.jl
#
# Loading CUDA (or AMDGPU/Metal) triggers the matching Allspark extension,
# which routes `sdpa` on device arrays to `flash_sdpa`. This checks that
# the device path produces the same result as the CPU materialized path.
using Test
using Random
using Allspark.Layers: sdpa, flash_sdpa

const GPU_BACKEND = get(ENV, "ALLSPARK_GPU_BACKEND", "cuda")

# Bring in the requested backend and return an `adapt`-style mover.
to_device, backend_name = if GPU_BACKEND == "cuda"
    using CUDA
    (CUDA.functional() ? CuArray : error("CUDA not functional")), "CUDA"
elseif GPU_BACKEND == "amdgpu"
    using AMDGPU
    ROCArray, "AMDGPU"
elseif GPU_BACKEND == "metal"
    using Metal
    MtlArray, "Metal"
else
    error("Unknown ALLSPARK_GPU_BACKEND=$(GPU_BACKEND); use cuda|amdgpu|metal")
end

function _qkv(sq, skv, d, batch; seed=0)
    Random.seed!(seed)
    return (
        randn(Float32, sq, d, batch),
        randn(Float32, d, skv, batch),
        randn(Float32, skv, d, batch),
    )
end

@testset verbose = true "GPU attention ($(backend_name))" begin
    scale = sqrt(8.0f0)

    @testset "device sdpa matches CPU materialized (no mask)" begin
        q, k_t, v = _qkv(4, 20, 8, 2; seed=0x10)
        cpu = sdpa(q, k_t, v; scale=scale)
        gpu = sdpa(to_device(q), to_device(k_t), to_device(v); scale=scale)
        @test Array(gpu) ≈ cpu rtol = 1e-3
    end

    @testset "device sdpa matches CPU (causal mask)" begin
        sq = skv = 16
        q, k_t, v = _qkv(sq, skv, 8, 2; seed=0x12)
        drop = Bool[(j - 1) > (i - 1) for i in 1:sq, j in 1:skv]
        cpu = sdpa(q, k_t, v; scale=scale, drop=drop)
        gpu = sdpa(
            to_device(q), to_device(k_t), to_device(v); scale=scale, drop=drop
        )
        @test Array(gpu) ≈ cpu rtol = 1e-3
    end

    @testset "device sdpa matches CPU (softcap + sliding window)" begin
        sq = skv = 12
        w = 4
        q, k_t, v = _qkv(sq, skv, 8, 1; seed=0x13)
        drop = Bool[
            (j - 1) > (i - 1) || (i - 1) - (j - 1) >= w for i in 1:sq, j in 1:skv
        ]
        cpu = sdpa(q, k_t, v; scale=scale, softcap=30.0f0, drop=drop)
        gpu = sdpa(
            to_device(q),
            to_device(k_t),
            to_device(v);
            scale=scale,
            softcap=30.0f0,
            drop=drop,
        )
        @test !any(isnan, Array(gpu))
        @test Array(gpu) ≈ cpu rtol = 1e-3
    end

    @testset "device sdpa actually dispatched to flash_sdpa" begin
        # If the extension didn't load, `sdpa` would fall through to the
        # materialized CPU method and error on device arrays inside
        # `batched_mul`/`softmax`. Reaching here at all is the check.
        q, k_t, v = _qkv(3, 5, 8, 1; seed=0x20)
        gpu = sdpa(to_device(q), to_device(k_t), to_device(v); scale=scale)
        @test gpu isa typeof(to_device(q))
    end
end
