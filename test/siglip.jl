using Test
using Random
using Flux
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models: siglip_state_dict_map, load_state_dict!

function _tiny_siglip_config()
    return SiglipConfig(;
        vision=SiglipVisionConfig(;
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_channels=3,
            image_size=8,
            patch_size=4,
            layer_norm_eps=1.0e-6,
        ),
        text=SiglipTextConfig(;
            vocab_size=20,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=6,
            projection_size=8,
            layer_norm_eps=1.0e-6,
        ),
    )
end

function _siglip_encoder_sd!(out, hf, n_layers, h, inter)
    for i in 0:(n_layers - 1)
        p = "$(hf).encoder.layers.$(i)"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj")
            out["$(p).self_attn.$(proj).weight"] = randn(Float32, h, h)
            out["$(p).self_attn.$(proj).bias"] = randn(Float32, h)
        end
        out["$(p).layer_norm1.weight"] = randn(Float32, h)
        out["$(p).layer_norm1.bias"] = randn(Float32, h)
        out["$(p).layer_norm2.weight"] = randn(Float32, h)
        out["$(p).layer_norm2.bias"] = randn(Float32, h)
        out["$(p).mlp.fc1.weight"] = randn(Float32, inter, h)
        out["$(p).mlp.fc1.bias"] = randn(Float32, inter)
        out["$(p).mlp.fc2.weight"] = randn(Float32, h, inter)
        out["$(p).mlp.fc2.bias"] = randn(Float32, h)
    end
end

function _siglip_sd(cfg::SiglipConfig)
    v, t = cfg.vision, cfg.text
    vh, th = v.hidden_size, t.hidden_size
    np = (v.image_size ÷ v.patch_size)^2
    out = Dict{String,Array{Float32}}()

    out["vision_model.embeddings.patch_embedding.weight"] = randn(
        Float32, vh, v.num_channels, v.patch_size, v.patch_size
    )
    out["vision_model.embeddings.patch_embedding.bias"] = randn(Float32, vh)
    out["vision_model.embeddings.position_embedding.weight"] = randn(Float32, np, vh)
    _siglip_encoder_sd!(out, "vision_model", v.num_hidden_layers, vh, v.intermediate_size)
    out["vision_model.post_layernorm.weight"] = randn(Float32, vh)
    out["vision_model.post_layernorm.bias"] = randn(Float32, vh)
    out["vision_model.head.probe"] = randn(Float32, 1, 1, vh)
    out["vision_model.head.attention.in_proj_weight"] = randn(Float32, 3vh, vh)
    out["vision_model.head.attention.in_proj_bias"] = randn(Float32, 3vh)
    out["vision_model.head.attention.out_proj.weight"] = randn(Float32, vh, vh)
    out["vision_model.head.attention.out_proj.bias"] = randn(Float32, vh)
    out["vision_model.head.layernorm.weight"] = randn(Float32, vh)
    out["vision_model.head.layernorm.bias"] = randn(Float32, vh)
    out["vision_model.head.mlp.fc1.weight"] = randn(Float32, v.intermediate_size, vh)
    out["vision_model.head.mlp.fc1.bias"] = randn(Float32, v.intermediate_size)
    out["vision_model.head.mlp.fc2.weight"] = randn(Float32, vh, v.intermediate_size)
    out["vision_model.head.mlp.fc2.bias"] = randn(Float32, vh)

    out["text_model.embeddings.token_embedding.weight"] = randn(Float32, t.vocab_size, th)
    out["text_model.embeddings.position_embedding.weight"] = randn(
        Float32, t.max_position_embeddings, th
    )
    _siglip_encoder_sd!(out, "text_model", t.num_hidden_layers, th, t.intermediate_size)
    out["text_model.final_layer_norm.weight"] = randn(Float32, th)
    out["text_model.final_layer_norm.bias"] = randn(Float32, th)
    out["text_model.head.weight"] = randn(Float32, t.projection_size, th)
    out["text_model.head.bias"] = randn(Float32, t.projection_size)

    out["logit_scale"] = Float32[0.3]
    out["logit_bias"] = Float32[-0.2]
    return out
end

# Explicit single-query multi-head attention using the head's own projections.
function _explicit_pool_attn(head, hs)
    hidden, np, N = size(hs)
    H, hd = head.num_heads, head.head_dim
    q = head.wq(head.probe)                        # (hidden,)
    pooled = zeros(Float32, hidden, N)
    for n in 1:N
        col = zeros(Float32, hidden)
        for hh in 1:H
            r = ((hh - 1) * hd + 1):(hh * hd)
            qh = q[r]
            scores = Float32[
                sum(qh .* head.wk(hs[:, k, n])[r]) / sqrt(Float32(hd)) for k in 1:np
            ]
            w = softmax(scores)
            acc = zeros(Float32, hd)
            for k in 1:np
                acc .+= w[k] .* head.wv(hs[:, k, n])[r]
            end
            col[r] = acc
        end
        pooled[:, n] = head.wo(col)
    end
    return pooled
end

@testset "SigLIP architecture" begin
    cfg = _tiny_siglip_config()
    m = SiglipModel(cfg)
    @test length(m.vision.layers) == cfg.vision.num_hidden_layers
    @test length(m.text.layers) == cfg.text.num_hidden_layers
    # No CLS: vision embeddings hold only patch projection + position table.
    @test fieldnames(typeof(m.vision.embeddings)) ==
        (:patch_embeddings, :position_embeddings)
    @test size(m.vision.embeddings.position_embeddings, 2) ==
        (cfg.vision.image_size ÷ cfg.vision.patch_size)^2
    for layer in m.vision.layers
        @test layer.self_attn.causal == false
        @test layer.self_attn.rope === nothing
        @test layer.mlp.approx == true               # gelu tanh
        @test layer.input_layernorm.eps ≈ 1.0f-6
    end
end

@testset "SigLIP forward shapes" begin
    Random.seed!(0xC1)
    cfg = _tiny_siglip_config()
    m = SiglipModel(cfg)
    px = randn(
        Float32, cfg.vision.num_channels, cfg.vision.image_size, cfg.vision.image_size, 2
    )
    ids = rand(0:(cfg.text.vocab_size - 1), 5, 3)

    @test size(m.vision(px)) == (cfg.vision.hidden_size, 2)
    @test size(m.text(ids)) == (cfg.text.projection_size, 3)
    @test size(m(px, ids)) == (2, 3)                 # (n_images, n_texts)
end

@testset "attention pooling head matches explicit reference" begin
    Random.seed!(0xC2)
    cfg = _tiny_siglip_config()
    m = SiglipModel(cfg)
    load_state_dict!(m, _siglip_sd(cfg))

    hs = randn(Float32, cfg.vision.hidden_size, 4, 2)
    head = m.vision.head
    attn = _explicit_pool_attn(head, hs)
    ref = attn .+ head.mlp(head.layernorm(attn))
    @test head(hs) ≈ ref
end

@testset "load_state_dict! populates SigLIP" begin
    Random.seed!(0xC3)
    cfg = _tiny_siglip_config()
    m = SiglipModel(cfg)
    sd = _siglip_sd(cfg)
    load_state_dict!(m, sd)
    vh = cfg.vision.hidden_size

    # Patch conv reshaped; vision position table transposed.
    @test m.vision.embeddings.patch_embeddings.projection.weight ==
        reshape(sd["vision_model.embeddings.patch_embedding.weight"], vh, :)
    @test m.vision.embeddings.position_embeddings ==
        permutedims(sd["vision_model.embeddings.position_embedding.weight"], (2, 1))

    # Pooling head: probe squeezed, fused in_proj split into wq/wk/wv.
    head = m.vision.head
    Wi = sd["vision_model.head.attention.in_proj_weight"]
    bi = sd["vision_model.head.attention.in_proj_bias"]
    @test head.probe == vec(sd["vision_model.head.probe"])
    @test head.wq.weight == Wi[1:vh, :]
    @test head.wk.weight == Wi[(vh + 1):(2vh), :]
    @test head.wv.weight == Wi[(2vh + 1):(3vh), :]
    @test head.wv.bias == bi[(2vh + 1):(3vh)]
    @test head.wo.weight == sd["vision_model.head.attention.out_proj.weight"]

    # A vision encoder layer + post_layernorm.
    l = m.vision.layers[1]
    @test l.self_attn.wq.weight ==
        sd["vision_model.encoder.layers.0.self_attn.q_proj.weight"]
    @test l.self_attn.wo.bias == sd["vision_model.encoder.layers.0.self_attn.out_proj.bias"]
    @test l.input_layernorm.weight == sd["vision_model.encoder.layers.0.layer_norm1.weight"]
    @test l.mlp.c_fc.weight == sd["vision_model.encoder.layers.0.mlp.fc1.weight"]
    @test m.vision.post_layernorm.bias == sd["vision_model.post_layernorm.bias"]

    # Text tower: transposed embeddings, a layer, final norm, head, scalars.
    @test m.text.token_embedding.weight ==
        permutedims(sd["text_model.embeddings.token_embedding.weight"], (2, 1))
    @test m.text.position_embeddings ==
        permutedims(sd["text_model.embeddings.position_embedding.weight"], (2, 1))
    @test m.text.layers[2].mlp.c_proj.weight ==
        sd["text_model.encoder.layers.1.mlp.fc2.weight"]
    @test m.text.final_layer_norm.weight == sd["text_model.final_layer_norm.weight"]
    @test m.text.head.weight == sd["text_model.head.weight"]
    @test m.logit_scale[1] == only(sd["logit_scale"])
    @test m.logit_bias[1] == only(sd["logit_bias"])
end

@testset "logit_bias shifts every logit; embeddings are unit-norm" begin
    Random.seed!(0xC4)
    cfg = _tiny_siglip_config()
    m = SiglipModel(cfg)
    load_state_dict!(m, _siglip_sd(cfg))
    px = randn(
        Float32, cfg.vision.num_channels, cfg.vision.image_size, cfg.vision.image_size, 2
    )
    ids = rand(0:(cfg.text.vocab_size - 1), 4, 3)

    l1 = m(px, ids)
    m.logit_bias[1] += 1.0f0
    l2 = m(px, ids)
    @test l2 ≈ l1 .+ 1.0f0

    img = m.vision(px)
    normed = HuggingFaceTransformers.Models.l2_normalize(img)
    @test all(isapprox.(vec(sqrt.(sum(abs2, normed; dims=1))), 1.0f0; atol=1.0e-5))
end

@testset "siglip_state_dict_map omits reshaped tensors" begin
    cfg = _tiny_siglip_config()
    mp = siglip_state_dict_map(cfg)
    @test haskey(mp, "vision_model.encoder.layers.0.self_attn.q_proj.weight")
    @test haskey(mp, "text_model.head.weight")
    @test mp["text_model.embeddings.token_embedding.weight"][2] == :transpose
    # Hand-loaded tensors are absent from the map.
    @test !haskey(mp, "vision_model.embeddings.patch_embedding.weight")
    @test !haskey(mp, "vision_model.head.probe")
    @test !haskey(mp, "vision_model.head.attention.in_proj_weight")
    @test !haskey(mp, "logit_scale")
end
