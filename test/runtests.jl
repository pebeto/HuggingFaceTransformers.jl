using Test
using Allspark

@testset verbose = true "Allspark.jl" begin
    @testset "loads cleanly" begin
        @test isdefined(Allspark, :HFHub)
        @test isdefined(Allspark, :Tokenizers)
        @test isdefined(Allspark, :Layers)
        @test isdefined(Allspark, :Models)
        @test isdefined(Allspark, :Generation)
    end

    @testset verbose = true "HFHub" begin
        include("hfhub.jl")
    end

    @testset verbose = true "Weights" begin
        include("weights.jl")
    end

    @testset verbose = true "Tokenizers" begin
        include("tokenizers.jl")
    end

    @testset verbose = true "Llama" begin
        include("llama.jl")
    end

    @testset verbose = true "Generation" begin
        include("generate.jl")
    end

    @testset verbose = true "ChatTemplate" begin
        include("chat_template.jl")
    end

    @testset verbose = true "Layers" begin
        include("layers.jl")
    end

    @testset "Aqua" begin
        include("aqua.jl")
    end

    @testset "JET smoke" begin
        include("jet.jl")
    end
end
