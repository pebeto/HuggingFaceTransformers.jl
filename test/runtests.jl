using Test
using Allspark

@testset verbose = true "Allspark.jl" begin
    @testset "loads cleanly" begin
        @test isdefined(Allspark, :HFHub)
        @test isdefined(Allspark, :GGUF)
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

    @testset verbose = true "GGUF" begin
        include("gguf.jl")
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

    @testset verbose = true "Phi3" begin
        include("phi3.jl")
    end

    @testset verbose = true "GPT2" begin
        include("gpt2.jl")
    end

    @testset verbose = true "NeoX" begin
        include("neox.jl")
    end

    @testset verbose = true "Bert" begin
        include("bert.jl")
    end

    @testset verbose = true "Embedding" begin
        include("embedding.jl")
    end

    @testset verbose = true "Nomic" begin
        include("nomic.jl")
    end

    @testset verbose = true "ViT" begin
        include("vit.jl")
    end

    @testset verbose = true "SigLIP" begin
        include("siglip.jl")
    end

    @testset verbose = true "DINOv2" begin
        include("dinov2.jl")
    end

    @testset verbose = true "Whisper" begin
        include("whisper.jl")
    end

    @testset verbose = true "LLaVA" begin
        include("llava.jl")
    end

    @testset verbose = true "Mixtral" begin
        include("mixtral_model.jl")
    end

    @testset verbose = true "Dtype" begin
        include("dtype.jl")
    end

    @testset verbose = true "Quantize" begin
        include("quantize.jl")
    end

    @testset verbose = true "LoRA" begin
        include("lora.jl")
    end

    @testset verbose = true "Checkpoint" begin
        include("checkpoint.jl")
    end

    @testset verbose = true "Generation" begin
        include("generate.jl")
    end

    @testset verbose = true "Speculative" begin
        include("speculative.jl")
    end

    @testset verbose = true "ChatTemplate" begin
        include("chat_template.jl")
    end

    @testset verbose = true "Layers" begin
        include("layers.jl")
    end

    @testset verbose = true "Attention" begin
        include("attention.jl")
    end

    @testset "Aqua" begin
        include("aqua.jl")
    end

    # JET smoke pass: gated because JET 0.11+ currently fails to precompile
    # on Julia 1.12 (upstream LoweredCodeUtils method-signature drift).
    # Set ALLSPARK_TEST_JET=1 once a compatible JET tag exists.
    if get(ENV, "ALLSPARK_TEST_JET", "0") == "1"
        @testset "JET smoke" begin
            include("jet.jl")
        end
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

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_PHI3", ""))
        @testset verbose = true "Phi3 parity" begin
            include("parity_phi3.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_GPT2", ""))
        @testset verbose = true "GPT2 parity" begin
            include("parity_gpt2.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_NEOX", ""))
        @testset verbose = true "NeoX parity" begin
            include("parity_neox.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_BERT", ""))
        @testset verbose = true "Bert parity" begin
            include("parity_bert.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_MIXTRAL", ""))
        @testset verbose = true "Mixtral parity" begin
            include("parity_mixtral.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_EMBEDDING", ""))
        @testset verbose = true "Embedding parity" begin
            include("parity_embedding.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_NOMIC", ""))
        @testset verbose = true "Nomic parity" begin
            include("parity_nomic.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_VIT", ""))
        @testset verbose = true "ViT parity" begin
            include("parity_vit.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_SIGLIP", ""))
        @testset verbose = true "SigLIP parity" begin
            include("parity_siglip.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_DINOV2", ""))
        @testset verbose = true "DINOv2 parity" begin
            include("parity_dinov2.jl")
        end
    end

    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_WHISPER", ""))
        @testset verbose = true "Whisper parity" begin
            include("parity_whisper.jl")
        end
    end

    # Tokenizer parity is family-agnostic: a generic harness over recorded
    # (prompt, ids, decoded) tuples from HuggingFace's `transformers`.
    #   ALLSPARK_TEST_PARITY_TOKENIZER=gpt2,bert-uncased  # subset
    #   ALLSPARK_TEST_PARITY_TOKENIZER=all                # every fixture present
    if !isempty(get(ENV, "ALLSPARK_TEST_PARITY_TOKENIZER", ""))
        @testset verbose = true "Tokenizer parity" begin
            include("parity_tokenizer.jl")
        end
    end
end
