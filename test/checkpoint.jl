using Test
using Random
using Flux
using HuggingFaceTransformers.Layers: checkpoint, Checkpointed, GeluMLP, Linear

@testset "checkpoint forward is transparent" begin
    Random.seed!(0x51)
    W = randn(Float32, 4, 3)
    x = randn(Float32, 3, 5)
    @test checkpoint(*, W, x) == W * x

    mlp = GeluMLP(6, 12; approx=false)
    z = randn(Float32, 6, 4)
    @test Checkpointed(mlp)(z) == mlp(z)
end

@testset "checkpoint gradient matches direct — input" begin
    Random.seed!(0x52)
    W = randn(Float32, 4, 3)
    x = randn(Float32, 3)
    g_direct = Flux.gradient(x -> sum(abs2, W * x), x)[1]
    g_ckpt = Flux.gradient(x -> sum(abs2, checkpoint(*, W, x)), x)[1]
    @test g_ckpt ≈ g_direct
end

@testset "checkpoint gradient matches direct — layer parameters" begin
    Random.seed!(0x53)
    mlp = GeluMLP(6, 12; approx=false)
    z = randn(Float32, 6, 4)
    g_direct = Flux.gradient(m -> sum(m(z)), mlp)[1]
    g_ckpt = Flux.gradient(m -> sum(checkpoint(m, z)), mlp)[1]
    @test g_ckpt.c_fc.weight ≈ g_direct.c_fc.weight
    @test g_ckpt.c_fc.bias ≈ g_direct.c_fc.bias
    @test g_ckpt.c_proj.weight ≈ g_direct.c_proj.weight
    @test g_ckpt.c_proj.bias ≈ g_direct.c_proj.bias
end

@testset "Checkpointed wrapper gradient matches the unwrapped layer" begin
    Random.seed!(0x54)
    mlp = GeluMLP(6, 12; approx=false)
    z = randn(Float32, 6, 4)
    g_direct = Flux.gradient(m -> sum(m(z)), mlp)[1]
    g_wrapped = Flux.gradient(c -> sum(c(z)), Checkpointed(mlp))[1]
    @test g_wrapped.layer.c_fc.weight ≈ g_direct.c_fc.weight
    @test g_wrapped.layer.c_proj.weight ≈ g_direct.c_proj.weight
end

@testset "stacking checkpointed layers matches a plain stack" begin
    Random.seed!(0x55)
    l1 = Linear(6, 6; bias=true)
    l2 = Linear(6, 6; bias=true)
    z = randn(Float32, 6, 4)

    plain(a, b) = sum(b(a(z)))
    ckpt(a, b) = sum(checkpoint(b, checkpoint(a, z)))
    g_direct = Flux.gradient(plain, l1, l2)
    g_ckpt = Flux.gradient(ckpt, l1, l2)
    @test g_ckpt[1].weight ≈ g_direct[1].weight
    @test g_ckpt[2].weight ≈ g_direct[2].weight
end
