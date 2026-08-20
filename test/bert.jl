using Test
using Random
using Flux
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models: bert_state_dict_map, load_state_dict!, load_into!

function _bert_synthetic_state_dict(cfg::BertConfig)
    out = Dict{String,Array{Float32}}()
    p = cfg.hf_prefix
    hp = cfg.head_prefix
    h = cfg.hidden_size
    inter = cfg.intermediate_size

    # Embeddings (HF stores (vocab, hidden)).
    out["$(p).embeddings.word_embeddings.weight"] = randn(Float32, cfg.vocab_size, h)
    out["$(p).embeddings.position_embeddings.weight"] = randn(
        Float32, cfg.max_position_embeddings + cfg.position_embedding_offset, h
    )
    out["$(p).embeddings.token_type_embeddings.weight"] = randn(
        Float32, cfg.type_vocab_size, h
    )
    out["$(p).embeddings.LayerNorm.weight"] = randn(Float32, h)
    out["$(p).embeddings.LayerNorm.bias"] = randn(Float32, h)

    for i in 0:(cfg.num_hidden_layers - 1)
        prefix = "$(p).encoder.layer.$(i)"
        out["$(prefix).attention.self.query.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.self.query.bias"] = randn(Float32, h)
        out["$(prefix).attention.self.key.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.self.key.bias"] = randn(Float32, h)
        out["$(prefix).attention.self.value.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.self.value.bias"] = randn(Float32, h)
        out["$(prefix).attention.output.dense.weight"] = randn(Float32, h, h)
        out["$(prefix).attention.output.dense.bias"] = randn(Float32, h)
        out["$(prefix).attention.output.LayerNorm.weight"] = randn(Float32, h)
        out["$(prefix).attention.output.LayerNorm.bias"] = randn(Float32, h)
        out["$(prefix).intermediate.dense.weight"] = randn(Float32, inter, h)
        out["$(prefix).intermediate.dense.bias"] = randn(Float32, inter)
        out["$(prefix).output.dense.weight"] = randn(Float32, h, inter)
        out["$(prefix).output.dense.bias"] = randn(Float32, h)
        out["$(prefix).output.LayerNorm.weight"] = randn(Float32, h)
        out["$(prefix).output.LayerNorm.bias"] = randn(Float32, h)
    end

    if cfg.head_prefix == "cls.predictions"
        out["$(hp).transform.dense.weight"] = randn(Float32, h, h)
        out["$(hp).transform.dense.bias"] = randn(Float32, h)
        out["$(hp).transform.LayerNorm.weight"] = randn(Float32, h)
        out["$(hp).transform.LayerNorm.bias"] = randn(Float32, h)
    else
        out["$(hp).dense.weight"] = randn(Float32, h, h)
        out["$(hp).dense.bias"] = randn(Float32, h)
        out["$(hp).layer_norm.weight"] = randn(Float32, h)
        out["$(hp).layer_norm.bias"] = randn(Float32, h)
    end
    out["$(hp).bias"] = randn(Float32, cfg.vocab_size)
    if !cfg.tie_word_embeddings
        out["$(hp).decoder.weight"] = randn(Float32, cfg.vocab_size, h)
    end
    return out
end

function _tiny_bert_config(; roberta::Bool=false)
    if roberta
        return BertConfig(;
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=14,         # +2 offset → 16 slots in the table
            type_vocab_size=1,
            layer_norm_eps=1.0e-5,
            pad_token_id=1,
            position_embedding_offset=2,
            hf_prefix="roberta",
            head_prefix="lm_head",
            tie_word_embeddings=true,
        )
    end
    return BertConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        type_vocab_size=2,
        layer_norm_eps=1.0e-12,
        pad_token_id=0,
        position_embedding_offset=0,
        hf_prefix="bert",
        head_prefix="cls.predictions",
        tie_word_embeddings=true,
    )
end

@testset verbose = true "BertForMaskedLM" begin
    Random.seed!(0x1234)
    cfg = _tiny_bert_config()
    lm = BertForMaskedLM(cfg)

    @test lm.config === cfg
    @test length(lm.trunk.encoder.layers) == cfg.num_hidden_layers

    @testset "encoder is bidirectional + post-norm + exact GELU" begin
        for layer in lm.trunk.encoder.layers
            @test layer.self_attn.causal == false       # bidirectional
            @test layer.self_attn.rope === nothing      # no RoPE
            # Post-norm: attn_norm and ffn_norm exist as LayerNorms (not RMSNorm).
            @test layer.attn_norm isa HuggingFaceTransformers.Layers.LayerNorm
            @test layer.ffn_norm isa HuggingFaceTransformers.Layers.LayerNorm
            # FFN linears carry bias.
            @test layer.intermediate.bias isa AbstractVector
            @test layer.output_dense.bias isa AbstractVector
        end
        # Embeddings LayerNorm.
        @test lm.trunk.embeddings.norm isa HuggingFaceTransformers.Layers.LayerNorm
    end

    @testset "forward shapes" begin
        ids = rand(0:(cfg.vocab_size - 1), 5, 2)
        logits = lm(ids)
        @test size(logits) == (cfg.vocab_size, 5, 2)
    end

    @testset "bidirectional attention: every query sees every key" begin
        # Gradient of output position 1 w.r.t. input position 4 must be non-zero
        # (under causal attention it would be exactly zero).
        ids = rand(0:(cfg.vocab_size - 1), 5, 1)
        # Differentiate through the trunk via embeddings — perturb a downstream
        # embedding column and check that an earlier token's output changes.
        e = copy(lm.trunk.embeddings.embed_tokens.weight)
        h0 = lm(ids)
        lm.trunk.embeddings.embed_tokens.weight .+= 0.01f0 * randn(Float32, size(e))
        h1 = lm(ids)
        # Restore.
        copyto!(lm.trunk.embeddings.embed_tokens.weight, e)
        @test h0 != h1
    end
end

@testset verbose = true "bert_state_dict_map" begin
    @testset "BERT prefix (cls.predictions)" begin
        cfg = _tiny_bert_config()
        sd_map = bert_state_dict_map(cfg)

        @test haskey(sd_map, "bert.embeddings.word_embeddings.weight")
        @test haskey(sd_map, "bert.embeddings.token_type_embeddings.weight")
        @test haskey(sd_map, "bert.encoder.layer.0.attention.self.query.weight")
        @test haskey(sd_map, "bert.encoder.layer.0.attention.output.LayerNorm.weight")
        @test haskey(sd_map, "bert.encoder.layer.0.intermediate.dense.weight")
        @test haskey(sd_map, "bert.encoder.layer.0.output.LayerNorm.weight")
        # Head under cls.predictions namespace.
        @test haskey(sd_map, "cls.predictions.transform.dense.weight")
        @test haskey(sd_map, "cls.predictions.transform.LayerNorm.weight")
        @test haskey(sd_map, "cls.predictions.bias")
        # Tied: no decoder.weight entry.
        @test !haskey(sd_map, "cls.predictions.decoder.weight")
        # All three embedding tables transpose; everything else as_is.
        @test sd_map["bert.embeddings.word_embeddings.weight"][2] == :transpose
        @test sd_map["bert.embeddings.position_embeddings.weight"][2] == :transpose
        @test sd_map["bert.embeddings.token_type_embeddings.weight"][2] == :transpose
        @test sd_map["bert.encoder.layer.0.attention.self.query.weight"][2] == :as_is
    end

    @testset "RoBERTa prefix (lm_head)" begin
        cfg = _tiny_bert_config(; roberta=true)
        sd_map = bert_state_dict_map(cfg)

        @test haskey(sd_map, "roberta.embeddings.word_embeddings.weight")
        @test haskey(sd_map, "roberta.encoder.layer.0.attention.self.query.weight")
        # RoBERTa-flavored head paths.
        @test haskey(sd_map, "lm_head.dense.weight")
        @test haskey(sd_map, "lm_head.layer_norm.weight")
        @test haskey(sd_map, "lm_head.bias")
        # No BERT-style nested transform.
        @test !haskey(sd_map, "lm_head.transform.dense.weight")
    end
end

@testset "load_state_dict! — BERT (tied)" begin
    Random.seed!(0x9999)
    cfg = _tiny_bert_config()
    lm = BertForMaskedLM(cfg)
    sd = _bert_synthetic_state_dict(cfg)
    @test !haskey(sd, "cls.predictions.decoder.weight")    # tied

    load_state_dict!(lm, sd)

    # Embeddings transposed.
    @test lm.trunk.embeddings.embed_tokens.weight ==
        permutedims(sd["bert.embeddings.word_embeddings.weight"], (2, 1))
    @test lm.trunk.embeddings.embed_positions.weight ==
        permutedims(sd["bert.embeddings.position_embeddings.weight"], (2, 1))
    @test lm.trunk.embeddings.embed_types.weight ==
        permutedims(sd["bert.embeddings.token_type_embeddings.weight"], (2, 1))
    # Embeddings LN.
    @test lm.trunk.embeddings.norm.weight == sd["bert.embeddings.LayerNorm.weight"]
    @test lm.trunk.embeddings.norm.bias == sd["bert.embeddings.LayerNorm.bias"]

    # Layer 0: attention + FFN + post-norms flow.
    @test lm.trunk.encoder.layers[1].self_attn.wq.weight ==
        sd["bert.encoder.layer.0.attention.self.query.weight"]
    @test lm.trunk.encoder.layers[1].self_attn.wq.bias ==
        sd["bert.encoder.layer.0.attention.self.query.bias"]
    @test lm.trunk.encoder.layers[1].self_attn.wo.bias ==
        sd["bert.encoder.layer.0.attention.output.dense.bias"]
    @test lm.trunk.encoder.layers[1].attn_norm.weight ==
        sd["bert.encoder.layer.0.attention.output.LayerNorm.weight"]
    @test lm.trunk.encoder.layers[1].intermediate.weight ==
        sd["bert.encoder.layer.0.intermediate.dense.weight"]
    @test lm.trunk.encoder.layers[1].output_dense.weight ==
        sd["bert.encoder.layer.0.output.dense.weight"]
    @test lm.trunk.encoder.layers[1].ffn_norm.bias ==
        sd["bert.encoder.layer.0.output.LayerNorm.bias"]

    # Head: tied decoder.weight equals the input embedding (HF stores it that way).
    @test lm.head.decoder.weight == sd["bert.embeddings.word_embeddings.weight"]
    @test lm.head.bias == sd["cls.predictions.bias"]
end

@testset "load_state_dict! — RoBERTa (lm_head naming)" begin
    Random.seed!(0xAAAA)
    cfg = _tiny_bert_config(; roberta=true)
    lm = BertForMaskedLM(cfg)
    sd = _bert_synthetic_state_dict(cfg)

    load_state_dict!(lm, sd)

    # RoBERTa-flavored head paths.
    @test lm.head.dense.weight == sd["lm_head.dense.weight"]
    @test lm.head.norm.weight == sd["lm_head.layer_norm.weight"]
    @test lm.head.bias == sd["lm_head.bias"]
    # Tied decoder.
    @test lm.head.decoder.weight == sd["roberta.embeddings.word_embeddings.weight"]
end

@testset "load_state_dict! — BERT untied head" begin
    Random.seed!(0xBEEF)
    cfg = BertConfig(;
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        type_vocab_size=2,
        tie_word_embeddings=false,
    )
    lm = BertForMaskedLM(cfg)
    sd = _bert_synthetic_state_dict(cfg)
    @test haskey(sd, "cls.predictions.decoder.weight")

    load_state_dict!(lm, sd)
    @test lm.head.decoder.weight == sd["cls.predictions.decoder.weight"]
end

@testset "renames legacy LayerNorm parameter names" begin
    # Checkpoints converted before transformers renamed LayerNorm's parameters
    # ship `LayerNorm.gamma` / `LayerNorm.beta`; `bert-base-uncased` still does.
    Random.seed!(0xB37)
    cfg = BertConfig(;
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=16, type_vocab_size=2,
    )
    modern = _bert_synthetic_state_dict(cfg)
    legacy = Dict(
        replace(replace(k, "LayerNorm.weight" => "LayerNorm.gamma"),
            "LayerNorm.bias" => "LayerNorm.beta") => v for (k, v) in modern
    )
    @test any(endswith("LayerNorm.gamma"), keys(legacy))   # fixture really is legacy

    ids = reshape([1, 5, 9, 2], :, 1)
    from_modern = load_state_dict!(BertForMaskedLM(cfg), modern)
    from_legacy = load_state_dict!(BertForMaskedLM(cfg), legacy)
    @test from_legacy(ids) ≈ from_modern(ids)

    # The rename is scoped to LayerNorm, so `gamma` elsewhere is left alone.
    other = Dict{String,Array{Float32}}("encoder.layer.0.layer_scale.gamma" => randn(Float32, 4))
    @test haskey(Models._bert_rename_legacy_norms(other), "encoder.layer.0.layer_scale.gamma")
end
