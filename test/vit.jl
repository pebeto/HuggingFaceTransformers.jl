using Test
using Random
using Flux
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models: vit_state_dict_map, load_state_dict!

function _tiny_vit_config()
    return ViTConfig(;
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=8,
        patch_size=4,
        num_channels=3,
        layer_norm_eps=1.0e-12,
        num_labels=5,
    )
end

_tiny_vit_num_pos(cfg) = (cfg.image_size ÷ cfg.patch_size)^2 + 1

function _vit_synthetic_sd(cfg::ViTConfig)
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    C, P = cfg.num_channels, cfg.patch_size
    num_pos = (cfg.image_size ÷ P)^2 + 1
    out = Dict{String,Array{Float32}}()
    out["vit.embeddings.cls_token"] = randn(Float32, 1, 1, h)
    out["vit.embeddings.position_embeddings"] = randn(Float32, 1, num_pos, h)
    out["vit.embeddings.patch_embeddings.projection.weight"] = randn(Float32, h, C, P, P)
    out["vit.embeddings.patch_embeddings.projection.bias"] = randn(Float32, h)
    for i in 0:(cfg.num_hidden_layers - 1)
        pre = "vit.encoder.layer.$(i)"
        for name in ("query", "key", "value")
            out["$(pre).attention.attention.$(name).weight"] = randn(Float32, h, h)
            out["$(pre).attention.attention.$(name).bias"] = randn(Float32, h)
        end
        out["$(pre).attention.output.dense.weight"] = randn(Float32, h, h)
        out["$(pre).attention.output.dense.bias"] = randn(Float32, h)
        out["$(pre).intermediate.dense.weight"] = randn(Float32, inter, h)
        out["$(pre).intermediate.dense.bias"] = randn(Float32, inter)
        out["$(pre).output.dense.weight"] = randn(Float32, h, inter)
        out["$(pre).output.dense.bias"] = randn(Float32, h)
        out["$(pre).layernorm_before.weight"] = randn(Float32, h)
        out["$(pre).layernorm_before.bias"] = randn(Float32, h)
        out["$(pre).layernorm_after.weight"] = randn(Float32, h)
        out["$(pre).layernorm_after.bias"] = randn(Float32, h)
    end
    out["vit.layernorm.weight"] = randn(Float32, h)
    out["vit.layernorm.bias"] = randn(Float32, h)
    out["classifier.weight"] = randn(Float32, cfg.num_labels, h)
    out["classifier.bias"] = randn(Float32, cfg.num_labels)
    return out
end

# Explicit conv reference: out[o, patch, n] = bias[o] + Σ_{c,i,j} W[o,c,i,j] *
# x[c, ih*P+i, iw*P+j, n], patch index = ih*gw + iw (width-patch fastest).
function _reference_patch_embed(conv_w, bias, x, P)
    hidden = size(conv_w, 1)
    C, H, W, N = size(x)
    gh, gw = H ÷ P, W ÷ P
    ref = zeros(Float32, hidden, gh * gw, N)
    for n in 1:N, ih in 0:(gh - 1), iw in 0:(gw - 1)
        patch = ih * gw + iw + 1
        for o in 1:hidden
            s = bias[o]
            for c in 1:C, i in 1:P, j in 1:P
                s += conv_w[o, c, i, j] * x[c, ih * P + i, iw * P + j, n]
            end
            ref[o, patch, n] = s
        end
    end
    return ref
end

@testset "ViT architecture" begin
    cfg = _tiny_vit_config()
    m = ViTForImageClassification(cfg)
    @test length(m.vit.layers) == cfg.num_hidden_layers
    for layer in m.vit.layers
        @test layer.self_attn.causal == false          # bidirectional
        @test layer.self_attn.rope === nothing          # no rotary
        @test layer.mlp isa HuggingFaceTransformers.Layers.GeluMLP
        @test layer.mlp.approx == false                 # exact GELU
    end
    @test size(m.vit.embeddings.position_embeddings, 2) == _tiny_vit_num_pos(cfg)
end

@testset "ViT forward shape" begin
    Random.seed!(0xD1)
    cfg = _tiny_vit_config()
    m = ViTForImageClassification(cfg)
    x = randn(Float32, cfg.num_channels, cfg.image_size, cfg.image_size, 2)
    logits = m(x)
    @test size(logits) == (cfg.num_labels, 2)
end

@testset "patch embedding equals explicit conv reference" begin
    Random.seed!(0xD2)
    cfg = _tiny_vit_config()
    m = ViTForImageClassification(cfg)
    sd = _vit_synthetic_sd(cfg)
    load_state_dict!(m, sd)

    x = randn(Float32, cfg.num_channels, cfg.image_size, cfg.image_size, 2)
    got = m.vit.embeddings.patch_embeddings(x)
    ref = _reference_patch_embed(
        sd["vit.embeddings.patch_embeddings.projection.weight"],
        sd["vit.embeddings.patch_embeddings.projection.bias"],
        x,
        cfg.patch_size,
    )
    @test got ≈ ref
end

@testset "load_state_dict! populates ViT" begin
    Random.seed!(0xD3)
    cfg = _tiny_vit_config()
    m = ViTForImageClassification(cfg)
    sd = _vit_synthetic_sd(cfg)
    load_state_dict!(m, sd)

    emb = m.vit.embeddings
    # Patch conv reshaped to the Linear; cls squeezed; pos squeezed + transposed.
    @test emb.patch_embeddings.projection.weight == reshape(
        sd["vit.embeddings.patch_embeddings.projection.weight"], cfg.hidden_size, :,
    )
    @test emb.cls_token == vec(sd["vit.embeddings.cls_token"])
    @test emb.position_embeddings ==
        permutedims(dropdims(sd["vit.embeddings.position_embeddings"]; dims=1), (2, 1))

    # A layer: q/k/v/o, both LayerNorms, and the GELU MLP.
    layer = m.vit.layers[1]
    @test layer.self_attn.wq.weight ==
        sd["vit.encoder.layer.0.attention.attention.query.weight"]
    @test layer.self_attn.wv.bias ==
        sd["vit.encoder.layer.0.attention.attention.value.bias"]
    @test layer.self_attn.wo.bias == sd["vit.encoder.layer.0.attention.output.dense.bias"]
    @test layer.input_layernorm.weight == sd["vit.encoder.layer.0.layernorm_before.weight"]
    @test layer.post_attention_layernorm.bias ==
        sd["vit.encoder.layer.0.layernorm_after.bias"]
    @test layer.mlp.c_fc.weight == sd["vit.encoder.layer.0.intermediate.dense.weight"]
    @test layer.mlp.c_proj.bias == sd["vit.encoder.layer.0.output.dense.bias"]
    @test m.vit.layers[2].mlp.c_proj.weight == sd["vit.encoder.layer.1.output.dense.weight"]

    @test m.vit.layernorm.weight == sd["vit.layernorm.weight"]
    @test m.classifier.weight == sd["classifier.weight"]
    @test m.classifier.bias == sd["classifier.bias"]
end

@testset "vit_state_dict_map omits reshaped tensors" begin
    cfg = _tiny_vit_config()
    mp = vit_state_dict_map(cfg)
    @test haskey(mp, "vit.encoder.layer.0.attention.attention.query.weight")
    @test haskey(mp, "vit.layernorm.weight")
    @test haskey(mp, "classifier.weight")
    # Patch conv / cls / position embeddings are reshaped by hand, not mapped.
    @test !haskey(mp, "vit.embeddings.patch_embeddings.projection.weight")
    @test !haskey(mp, "vit.embeddings.cls_token")
    @test !haskey(mp, "vit.embeddings.position_embeddings")
end
