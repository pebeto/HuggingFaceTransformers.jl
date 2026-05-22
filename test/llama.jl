using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models:
    TokenEmbedding, build_caches, llama_state_dict_map, load_state_dict!, load_into!

function _synthetic_state_dict(cfg::LlamaConfig)
    out = Dict{String,Array{Float32}}()
    n_q = cfg.num_attention_heads * cfg.head_dim
    n_kv = cfg.num_key_value_heads * cfg.head_dim

    out["model.embed_tokens.weight"] = randn(Float32, cfg.vocab_size, cfg.hidden_size)
    out["model.norm.weight"] = randn(Float32, cfg.hidden_size)
    if !cfg.tie_word_embeddings
        out["lm_head.weight"] = randn(Float32, cfg.vocab_size, cfg.hidden_size)
    end

    for i in 0:(cfg.num_hidden_layers - 1)
        p = "model.layers.$(i)"
        out["$(p).input_layernorm.weight"] = randn(Float32, cfg.hidden_size)
        out["$(p).post_attention_layernorm.weight"] = randn(Float32, cfg.hidden_size)
        out["$(p).self_attn.q_proj.weight"] = randn(Float32, n_q, cfg.hidden_size)
        out["$(p).self_attn.k_proj.weight"] = randn(Float32, n_kv, cfg.hidden_size)
        out["$(p).self_attn.v_proj.weight"] = randn(Float32, n_kv, cfg.hidden_size)
        out["$(p).self_attn.o_proj.weight"] = randn(Float32, cfg.hidden_size, n_q)
        out["$(p).mlp.gate_proj.weight"] =
            randn(Float32, cfg.intermediate_size, cfg.hidden_size)
        out["$(p).mlp.up_proj.weight"] =
            randn(Float32, cfg.intermediate_size, cfg.hidden_size)
        out["$(p).mlp.down_proj.weight"] =
            randn(Float32, cfg.hidden_size, cfg.intermediate_size)
    end
    return out
end

function _tiny_config(; rope_scaling=nothing)
    return LlamaConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=10000.0,
        rms_norm_eps=1.0e-5,
        rope_scaling=rope_scaling,
    )
end

@testset "TokenEmbedding" begin
    embed = TokenEmbedding(10, 4)
    @test size(embed.weight) == (4, 10)

    ids_vec = [0, 3, 9]
    @test size(embed(ids_vec)) == (4, 3)

    ids_mat = [0 1; 2 3; 4 5]
    out = embed(ids_mat)
    @test size(out) == (4, 3, 2)

    # Embeddings are looked up correctly: column for id == column of weight at id+1
    @test embed([0])[:, 1] == embed.weight[:, 1]
    @test embed([7])[:, 1] == embed.weight[:, 8]
end

@testset "LlamaForCausalLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_config()
    lm = LlamaForCausalLM(cfg)

    @test lm.config === cfg
    @test length(lm.model.layers) == cfg.num_hidden_layers
    @test size(lm.lm_head.weight) == (cfg.vocab_size, cfg.hidden_size)

    @testset "forward shapes" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        logits = lm(ids)
        @test size(logits) == (cfg.vocab_size, 5, 2)
    end

    @testset "gradients flow" begin
        ids = rand(0:(cfg.vocab_size - 1), 3, 1)
        grads = Flux.gradient(m -> sum(m(ids)), lm)
        @test grads[1] !== nothing
        @test grads[1].model !== nothing
        @test grads[1].lm_head !== nothing
    end

    @testset "KV-cache prefill→decode parity" begin
        ids = rand(0:(cfg.vocab_size - 1), 4, 1)
        out_full = lm(ids)

        caches = build_caches(lm, cfg.max_position_embeddings, 1)
        @test length(caches) == cfg.num_hidden_layers
        @test size(caches[1].k) ==
            (cfg.head_dim, cfg.num_key_value_heads, cfg.max_position_embeddings, 1)

        cache0 = caches[1]
        k_arr_id = objectid(cache0.k)

        out_prefill = lm(ids[1:3, :]; caches=caches, step=1)
        @test size(out_prefill) == (cfg.vocab_size, 3, 1)

        # The cache is the *same* allocation — we mutate it, never replace it.
        @test caches[1] === cache0
        @test objectid(cache0.k) === k_arr_id
        @test any(cache0.k[:, :, 1:3, :] .!= 0)
        @test all(cache0.k[:, :, 4:end, :] .== 0)

        out_decode = lm(ids[4:4, :]; caches=caches, step=4)
        @test size(out_decode) == (cfg.vocab_size, 1, 1)

        @test isapprox(out_full[:, 4:4, :], out_decode; atol=1.0e-4)
    end
end

@testset "llama_state_dict_map" begin
    cfg = _tiny_config()
    mapping = llama_state_dict_map(cfg)

    # Mapping contains every per-layer key for every layer.
    per_layer_suffixes = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    expected_keys = String[]
    push!(expected_keys, "model.embed_tokens.weight", "model.norm.weight")
    push!(expected_keys, "lm_head.weight")
    for i in 0:(cfg.num_hidden_layers - 1), suf in per_layer_suffixes
        push!(expected_keys, "model.layers.$(i).$(suf)")
    end
    @test sort(collect(keys(mapping))) == sort(expected_keys)

    # Embedding is the only entry that needs a transpose.
    @test mapping["model.embed_tokens.weight"][2] === :transpose
    for k in keys(mapping)
        k == "model.embed_tokens.weight" && continue
        @test mapping[k][2] === :as_is
    end

    # Tied models drop the lm_head entry.
    cfg_tied = LlamaConfig(;
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        tie_word_embeddings=true,
    )
    @test !haskey(llama_state_dict_map(cfg_tied), "lm_head.weight")
end

@testset "load_state_dict! — untied" begin
    Random.seed!(0x4242)
    cfg = _tiny_config()
    lm = LlamaForCausalLM(cfg)
    sd = _synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    # Embed weight is the HF tensor transposed.
    @test lm.model.embed_tokens.weight == permutedims(sd["model.embed_tokens.weight"], (2, 1))
    @test lm.model.norm.weight == sd["model.norm.weight"]
    @test lm.lm_head.weight == sd["lm_head.weight"]

    # Spot-check one decoder layer end-to-end.
    layer = lm.model.layers[1]
    @test layer.input_layernorm.weight == sd["model.layers.0.input_layernorm.weight"]
    @test layer.self_attn.wq.weight == sd["model.layers.0.self_attn.q_proj.weight"]
    @test layer.self_attn.wk.weight == sd["model.layers.0.self_attn.k_proj.weight"]
    @test layer.self_attn.wv.weight == sd["model.layers.0.self_attn.v_proj.weight"]
    @test layer.self_attn.wo.weight == sd["model.layers.0.self_attn.o_proj.weight"]
    @test layer.post_attention_layernorm.weight ==
        sd["model.layers.0.post_attention_layernorm.weight"]
    @test layer.mlp.gate_proj.weight == sd["model.layers.0.mlp.gate_proj.weight"]
    @test layer.mlp.up_proj.weight == sd["model.layers.0.mlp.up_proj.weight"]
    @test layer.mlp.down_proj.weight == sd["model.layers.0.mlp.down_proj.weight"]

    # Forward still works after loading.
    ids = rand(0:(cfg.vocab_size - 1), 3, 1)
    @test size(lm(ids)) == (cfg.vocab_size, 3, 1)
end

@testset "load_state_dict! — tied embeddings" begin
    Random.seed!(0x5151)
    cfg = LlamaConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        tie_word_embeddings=true,
    )
    lm = LlamaForCausalLM(cfg)
    sd = _synthetic_state_dict(cfg)
    @test !haskey(sd, "lm_head.weight")

    load_state_dict!(lm, sd)

    # lm_head ends up as a copy of the (already-transposed-on-load) embedding,
    # transposed back so it matches HF's lm_head storage. End-to-end:
    # lm_head.weight should equal the HF embedding tensor itself.
    @test lm.lm_head.weight == sd["model.embed_tokens.weight"]
end

@testset "load_state_dict! — error paths" begin
    cfg = _tiny_config()
    lm = LlamaForCausalLM(cfg)

    @testset "missing key" begin
        sd = _synthetic_state_dict(cfg)
        delete!(sd, "model.norm.weight")
        @test_throws ArgumentError load_state_dict!(lm, sd)
    end

    @testset "shape mismatch" begin
        sd = _synthetic_state_dict(cfg)
        sd["model.norm.weight"] = randn(Float32, cfg.hidden_size + 1)
        @test_throws DimensionMismatch load_state_dict!(lm, sd)
    end

    @testset "extra keys are ignored" begin
        sd = _synthetic_state_dict(cfg)
        sd["model.layers.0.fictional.bias"] = randn(Float32, 7)
        @test load_state_dict!(lm, sd) === lm
    end
end

@testset "Llama-3 rope scaling wires through" begin
    cfg = _tiny_config(;
        rope_scaling=LlamaRopeScaling(;
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8,
        ),
    )
    lm = LlamaForCausalLM(cfg)
    rope_scaled = lm.model.layers[1].self_attn.rope
    rope_unscaled = LlamaForCausalLM(_tiny_config()).model.layers[1].self_attn.rope
    @test rope_scaled.inv_freq != rope_unscaled.inv_freq

    ids = rand(0:(cfg.vocab_size - 1), 3, 1)
    @test size(lm(ids)) == (cfg.vocab_size, 3, 1)
end
