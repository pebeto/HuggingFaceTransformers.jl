using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: build_caches, mixtral_state_dict_map, load_state_dict!, load_into!
using Allspark.Layers: MoEMLP

function _mixtral_synthetic_state_dict(cfg::MixtralConfig)
    out = Dict{String,Array{Float32}}()
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    n_q = cfg.num_attention_heads * cfg.head_dim
    n_kv = cfg.num_key_value_heads * cfg.head_dim

    out["model.embed_tokens.weight"] = randn(Float32, cfg.vocab_size, h)
    out["model.norm.weight"] = randn(Float32, h)
    if !cfg.tie_word_embeddings
        out["lm_head.weight"] = randn(Float32, cfg.vocab_size, h)
    end

    for i in 0:(cfg.num_hidden_layers - 1)
        p = "model.layers.$(i)"
        out["$(p).input_layernorm.weight"] = randn(Float32, h)
        out["$(p).post_attention_layernorm.weight"] = randn(Float32, h)
        out["$(p).self_attn.q_proj.weight"] = randn(Float32, n_q, h)
        out["$(p).self_attn.k_proj.weight"] = randn(Float32, n_kv, h)
        out["$(p).self_attn.v_proj.weight"] = randn(Float32, n_kv, h)
        out["$(p).self_attn.o_proj.weight"] = randn(Float32, h, n_q)

        # Router (gate).
        out["$(p).block_sparse_moe.gate.weight"] = randn(Float32, cfg.num_local_experts, h)
        # Experts. HF's w1/w2/w3 = gate_proj/down_proj/up_proj.
        for e in 0:(cfg.num_local_experts - 1)
            pe = "$(p).block_sparse_moe.experts.$(e)"
            out["$(pe).w1.weight"] = randn(Float32, inter, h)   # gate_proj
            out["$(pe).w2.weight"] = randn(Float32, h, inter)   # down_proj
            out["$(pe).w3.weight"] = randn(Float32, inter, h)   # up_proj
        end
    end
    return out
end

function _tiny_mixtral_config(; num_local_experts=4, num_experts_per_tok=2)
    return MixtralConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=1_000_000.0,
        rms_norm_eps=1.0e-5,
        tie_word_embeddings=false,
        sliding_window=nothing,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
end

@testset verbose = true "MixtralForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_mixtral_config()
    lm = MixtralForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers

    @testset "every decoder layer holds an MoE MLP" begin
        for layer in lm.model.layers
            @test layer.mlp isa Allspark.Layers.MoEMLP
            @test length(layer.mlp.experts) == cfg.num_local_experts
            @test layer.mlp.num_experts == cfg.num_local_experts
            @test layer.mlp.top_k == cfg.num_experts_per_tok
            # The gate is a bias-less projection from hidden to num_experts.
            @test size(layer.mlp.gate.weight) == (cfg.num_local_experts, cfg.hidden_size)
            @test layer.mlp.gate.bias === nothing
        end
    end

    @testset "forward shapes" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        logits = lm(ids)
        @test size(logits) == (cfg.vocab_size, 5, 2)
    end

    @testset "KV-cache prefill→decode parity" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        out_full = lm(ids)
        caches = build_caches(lm, cfg.max_position_embeddings, 1)
        out_prefill = lm(ids[1:3, :]; caches=caches, step=1)
        out_decode = lm(ids[4:4, :]; caches=caches, step=4)
        @test isapprox(out_full[:, 4:4, :], out_decode; atol=1.0e-4)
    end
end

@testset verbose = true "MoEMLP routing math" begin
    Random.seed!(0xC0FFEE)
    hidden = 8
    inter = 16
    n_experts = 4

    @testset "top-K = num_experts is the weighted ensemble" begin
        # When all experts are picked, the renormalization just rescales by
        # 1 / sum(probs) = 1 (since softmax sums to 1). So output equals
        # the per-token probability-weighted sum of every expert's output.
        moe = MoEMLP(hidden, inter, n_experts, n_experts)
        x = randn(Float32, hidden, 5, 1)
        out = moe(x)

        # Reference: same routing, manual computation.
        flat = reshape(x, hidden, 5)
        probs = Flux.softmax(moe.gate(flat); dims=1)
        expected = zeros(Float32, hidden, 5)
        for e in 1:n_experts
            expected .+= reshape(probs[e, :], 1, 5) .* moe.experts[e](flat)
        end
        @test reshape(out, hidden, 5) ≈ expected
    end

    @testset "top-K < num_experts: only the top-K experts contribute" begin
        moe = MoEMLP(hidden, inter, n_experts, 2)
        # Positive-only inputs so the gate logits keep their sign — the
        # constant gate rows below would flip otherwise.
        x = rand(Float32, hidden, 3, 1)

        # Forge the router weights so every token picks experts 1 and 2.
        # Set gate.weight rows so expert 1 and 2 get the largest scores.
        moe.gate.weight .= 0
        moe.gate.weight[1, :] .= 10.0f0       # expert 1 always wins
        moe.gate.weight[2, :] .= 5.0f0        # expert 2 always second

        out = moe(x)

        # Run again with experts 3 and 4's weights perturbed — output must
        # not change, since neither is selected for any token.
        snapshot = (
            copy(moe.experts[3].gate_proj.weight),
            copy(moe.experts[4].down_proj.weight),
        )
        moe.experts[3].gate_proj.weight .+= 100.0f0 .* randn(Float32, size(snapshot[1]))
        moe.experts[4].down_proj.weight .+= 100.0f0 .* randn(Float32, size(snapshot[2]))
        out2 = moe(x)
        @test out ≈ out2
        # Restore.
        moe.experts[3].gate_proj.weight .= snapshot[1]
        moe.experts[4].down_proj.weight .= snapshot[2]
    end

    @testset "routing weights sum to 1 per token (after renormalization)" begin
        # The forward doesn't expose routing weights directly; verify by
        # constructing a deterministic case. Force expert-1 win across all
        # tokens with top_k=1: output for each token must equal the
        # expert-1 output exactly (weight = 1).
        moe = MoEMLP(hidden, inter, n_experts, 1)
        moe.gate.weight .= 0
        moe.gate.weight[1, :] .= 10.0f0   # expert 1 always wins
        # Positive-only inputs so expert 1's score stays largest.
        x = rand(Float32, hidden, 4, 1)
        out = moe(x)
        flat = reshape(x, hidden, 4)
        expected = moe.experts[1](flat)
        @test reshape(out, hidden, 4) ≈ expected
    end

    @testset "different routing gives different outputs" begin
        moe = MoEMLP(hidden, inter, n_experts, 2)
        x = randn(Float32, hidden, 5, 1)
        out_default = moe(x)
        # Bias the router toward different experts.
        moe.gate.weight .= 0
        moe.gate.weight[3, :] .= 10.0f0
        moe.gate.weight[4, :] .= 5.0f0
        out_skewed = moe(x)
        @test !(out_default ≈ out_skewed)
    end
end

@testset "mixtral_state_dict_map" begin
    cfg = _tiny_mixtral_config()
    sd_map = mixtral_state_dict_map(cfg)

    # Per-layer entries (attention + norms + MoE).
    @test haskey(sd_map, "model.embed_tokens.weight")
    @test haskey(sd_map, "model.norm.weight")
    @test haskey(sd_map, "lm_head.weight")          # untied
    @test haskey(sd_map, "model.layers.0.self_attn.q_proj.weight")
    @test haskey(sd_map, "model.layers.0.block_sparse_moe.gate.weight")
    # All experts present.
    for e in 0:(cfg.num_local_experts - 1)
        @test haskey(sd_map, "model.layers.0.block_sparse_moe.experts.$(e).w1.weight")
        @test haskey(sd_map, "model.layers.0.block_sparse_moe.experts.$(e).w2.weight")
        @test haskey(sd_map, "model.layers.0.block_sparse_moe.experts.$(e).w3.weight")
    end

    # Entry count: embed + norm + lm_head + per-layer (4 attn + 2 norms + 1 gate + 3 per expert).
    expected_per_layer = 4 + 2 + 1 + 3 * cfg.num_local_experts
    @test length(sd_map) == 3 + cfg.num_hidden_layers * expected_per_layer
end

@testset "load_state_dict! — Mixtral (HF w1/w2/w3 → gate/down/up_proj)" begin
    Random.seed!(0x9999)
    cfg = _tiny_mixtral_config()
    lm = MixtralForCausalLM(cfg)
    sd = _mixtral_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    # Layer 0 spot checks.
    @test lm.model.layers[1].self_attn.wq.weight ==
        sd["model.layers.0.self_attn.q_proj.weight"]
    @test lm.model.layers[1].mlp.gate.weight ==
        sd["model.layers.0.block_sparse_moe.gate.weight"]

    # Each expert's w1/w2/w3 must land in the right slot. The wN/proj map is
    # the easy thing to get wrong: w2 is the down-projection, not the
    # up-projection.
    for e in 0:(cfg.num_local_experts - 1)
        expert = lm.model.layers[1].mlp.experts[e + 1]
        @test expert.gate_proj.weight ==
            sd["model.layers.0.block_sparse_moe.experts.$(e).w1.weight"]
        @test expert.down_proj.weight ==
            sd["model.layers.0.block_sparse_moe.experts.$(e).w2.weight"]
        @test expert.up_proj.weight ==
            sd["model.layers.0.block_sparse_moe.experts.$(e).w3.weight"]
    end

    @test lm.lm_head.weight == sd["lm_head.weight"]
end
