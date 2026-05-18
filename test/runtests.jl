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

    @testset "Aqua" begin
        include("aqua.jl")
    end

    @testset "JET smoke" begin
        include("jet.jl")
    end
end
