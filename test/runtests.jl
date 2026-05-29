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

    @testset verbose = true "Mistral" begin
        include("mistral.jl")
    end

    @testset verbose = true "Qwen" begin
        include("qwen.jl")
    end

    @testset verbose = true "Gemma" begin
        include("gemma.jl")
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

    # Integration parity tests: each variant needs an HF model download +
    # a recorded Python-side fixture. Gated so default `Pkg.test()` stays
    # fast and offline. Examples:
    #   ALLSPARK_TEST_PARITY=1B           # one variant
    #   ALLSPARK_TEST_PARITY=1B,3B        # several
    #   ALLSPARK_TEST_PARITY=all          # everything with a fixture present
    #   ALLSPARK_TEST_PARITY=1            # legacy alias for 1B
    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY", ""))
        @testset verbose = true "Llama parity" begin
            include("parity_llama.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_MISTRAL", ""))
        @testset verbose = true "Mistral parity" begin
            include("parity_mistral.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_QWEN", ""))
        @testset verbose = true "Qwen parity" begin
            include("parity_qwen.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_GEMMA", ""))
        @testset verbose = true "Gemma parity" begin
            include("parity_gemma.jl")
        end
    end
end
