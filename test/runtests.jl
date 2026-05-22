using Test
using Allspark

@testset "Allspark.jl" begin
    @testset "loads cleanly" begin
        @test isdefined(Allspark, :HFHub)
        @test isdefined(Allspark, :Tokenizers)
        @test isdefined(Allspark, :Layers)
        @test isdefined(Allspark, :Models)
        @test isdefined(Allspark, :Generation)
    end

    @testset "HFHub" begin
        include("hfhub.jl")
    end

    @testset "Weights" begin
        include("weights.jl")
    end

    @testset "Tokenizers" begin
        include("tokenizers.jl")
    end

    @testset "Llama" begin
        include("llama.jl")
    end

    @testset "Generation" begin
        include("generate.jl")
    end

    @testset "Layers" begin
        include("layers.jl")
    end

    @testset "Aqua" begin
        include("aqua.jl")
    end

    @testset "JET smoke" begin
        include("jet.jl")
    end
end
