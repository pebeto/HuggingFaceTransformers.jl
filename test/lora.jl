using Test
using Random
using Flux
using SafeTensors: load_safetensors
using Allspark.Models
using Allspark.Models: llama_state_dict_map, load_state_dict!
using Allspark.Layers: Linear

@testset "LoRALinear forward math (2-D and N-D)" begin
    Random.seed!(0xC1)
    base = Linear(6, 8; bias=true)
    copyto!(base.weight, randn(Float32, 8, 6))
    copyto!(base.bias, randn(Float32, 8))
    A = randn(Float32, 4, 6)
    B = randn(Float32, 8, 4)
    l = LoRALinear(base, A, B, 2.0f0)

    x = randn(Float32, 6, 3)
    @test l(x) ≈ base(x) .+ 2.0f0 .* (B * (A * x))

    x3 = randn(Float32, 6, 3, 2)
    expected = base(x3) .+ 2.0f0 .* reshape(B * (A * reshape(x3, 6, 6)), 8, 3, 2)
    @test l(x3) ≈ expected
end

@testset "lora_wrap starts as a no-op (B = 0)" begin
    Random.seed!(0xC2)
    base = Linear(6, 8; bias=true)
    copyto!(base.weight, randn(Float32, 8, 6))
    lw = lora_wrap(base, 4, 8.0)
    @test lw.scaling == 8.0f0 / 4
    @test all(lw.lora_B .== 0)
    x = randn(Float32, 6, 3)
    @test lw(x) ≈ base(x)                 # zero B ⇒ identity update
end

@testset "merge_lora folds the update into a Linear" begin
    Random.seed!(0xC3)
    base = Linear(6, 8; bias=true)
    copyto!(base.weight, randn(Float32, 8, 6))
    copyto!(base.bias, randn(Float32, 8))
    A = randn(Float32, 4, 6)
    B = randn(Float32, 8, 4)
    l = LoRALinear(base, A, B, 2.0f0)
    merged = merge_lora(l)

    @test merged isa Linear
    @test merged.weight ≈ base.weight .+ 2.0f0 .* (B * A)
    x = randn(Float32, 6, 3)
    @test merged(x) ≈ l(x)
end

@testset "lora_scaling: standard vs rsLoRA" begin
    @test lora_scaling(LoraConfig(; r=4, lora_alpha=8.0, target_modules=["q"])) == 2.0f0
    @test lora_scaling(LoraConfig(; r=4, lora_alpha=8.0, target_modules=["q"], use_rslora=true)) ≈
        8.0f0 / sqrt(4)
end

@testset "safetensors writer round-trips through the reader" begin
    Random.seed!(0xC4)
    mktempdir() do dir
        path = joinpath(dir, "t.safetensors")
        ts = Dict(
            "w1d" => randn(Float32, 7),
            "w2d" => randn(Float32, 3, 5),
            "w3d" => randn(Float32, 2, 3, 4),
        )
        Allspark.Models._save_safetensors(path, ts)
        back = load_safetensors(path; mmap=false)
        for (k, v) in ts
            @test size(back[k]) == size(v)
            @test back[k] ≈ v
        end
    end
end

@testset "save_lora / load_lora PEFT round-trip" begin
    Random.seed!(0xC5)
    mktempdir() do dir
        cfg = LoraConfig(; r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
        mw = Dict(
            "model.layers.0.self_attn.q_proj" => (randn(Float32, 4, 16), randn(Float32, 16, 4)),
            "model.layers.0.self_attn.v_proj" => (randn(Float32, 4, 16), randn(Float32, 16, 4)),
        )
        save_lora(dir, cfg, mw)
        @test isfile(joinpath(dir, "adapter_config.json"))
        @test isfile(joinpath(dir, "adapter_model.safetensors"))

        cfg2, w2 = load_lora(dir)
        @test cfg2.r == 4
        @test cfg2.lora_alpha == 8.0
        @test cfg2.target_modules == ["q_proj", "v_proj"]

        kA = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        kB = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        @test haskey(w2, kA) && haskey(w2, kB)
        @test w2[kA] ≈ mw["model.layers.0.self_attn.q_proj"][1]
        @test w2[kB] ≈ mw["model.layers.0.self_attn.q_proj"][2]
    end
end

@testset "apply_lora! merges the adapter into a Llama" begin
    Random.seed!(0xC6)
    cfg_llm = LlamaConfig(;
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=32,
        rope_theta=10000.0,
        tie_word_embeddings=false,
    )
    lm = LlamaForCausalLM(cfg_llm)
    orig_q0 = copy(lm.model.layers[1].self_attn.wq.weight)
    orig_o0 = copy(lm.model.layers[1].self_attn.wo.weight)   # untargeted

    mw = Dict{String,Tuple{Matrix{Float32},Matrix{Float32}}}()
    for i in 0:1, proj in ("q_proj", "v_proj")
        mw["model.layers.$(i).self_attn.$(proj)"] = (randn(Float32, 4, 16), randn(Float32, 16, 4))
    end
    cfg = LoraConfig(; r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])

    mktempdir() do dir
        save_lora(dir, cfg, mw)
        applied = apply_lora!(lm, dir, llama_state_dict_map(cfg_llm))
        @test applied == 4

        A, B = mw["model.layers.0.self_attn.q_proj"]
        @test lm.model.layers[1].self_attn.wq.weight ≈ orig_q0 .+ 2.0f0 .* (B * A)
        @test lm.model.layers[1].self_attn.wo.weight == orig_o0   # untargeted unchanged
    end
end

@testset "apply_lora! rejects an unknown target module" begin
    Random.seed!(0xC7)
    cfg_llm = LlamaConfig(;
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, head_dim=4,
        max_position_embeddings=32, rope_theta=10000.0,
    )
    lm = LlamaForCausalLM(cfg_llm)
    mktempdir() do dir
        save_lora(
            dir,
            LoraConfig(; r=2, lora_alpha=4.0, target_modules=["bogus"]),
            Dict("model.layers.0.self_attn.bogus" => (randn(Float32, 2, 16), randn(Float32, 16, 2))),
        )
        @test_throws ArgumentError apply_lora!(lm, dir, llama_state_dict_map(cfg_llm))
    end
end
