using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: dinov2_state_dict_map, load_state_dict!

function _tiny_dinov2_config(; num_register_tokens=2)
    return Dinov2Config(;
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=8,
        patch_size=4,
        num_channels=3,
        layer_norm_eps=1.0e-6,
        layerscale_value=1.0,
        num_register_tokens=num_register_tokens,
    )
end

function _dinov2_synthetic_sd(cfg::Dinov2Config)
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    C, P = cfg.num_channels, cfg.patch_size
    num_pos = (cfg.image_size ÷ P)^2 + 1
    out = Dict{String,Array{Float32}}()
    out["embeddings.cls_token"] = randn(Float32, 1, 1, h)
    out["embeddings.position_embeddings"] = randn(Float32, 1, num_pos, h)
    out["embeddings.patch_embeddings.projection.weight"] = randn(Float32, h, C, P, P)
    out["embeddings.patch_embeddings.projection.bias"] = randn(Float32, h)
    if cfg.num_register_tokens > 0
        out["embeddings.register_tokens"] = randn(Float32, 1, cfg.num_register_tokens, h)
    end
    for i in 0:(cfg.num_hidden_layers - 1)
        pre = "encoder.layer.$(i)"
        out["$(pre).norm1.weight"] = randn(Float32, h)
        out["$(pre).norm1.bias"] = randn(Float32, h)
        for name in ("query", "key", "value")
            out["$(pre).attention.attention.$(name).weight"] = randn(Float32, h, h)
            out["$(pre).attention.attention.$(name).bias"] = randn(Float32, h)
        end
        out["$(pre).attention.output.dense.weight"] = randn(Float32, h, h)
        out["$(pre).attention.output.dense.bias"] = randn(Float32, h)
        out["$(pre).layer_scale1.lambda1"] = randn(Float32, h)
        out["$(pre).norm2.weight"] = randn(Float32, h)
        out["$(pre).norm2.bias"] = randn(Float32, h)
        out["$(pre).mlp.fc1.weight"] = randn(Float32, inter, h)
        out["$(pre).mlp.fc1.bias"] = randn(Float32, inter)
        out["$(pre).mlp.fc2.weight"] = randn(Float32, h, inter)
        out["$(pre).mlp.fc2.bias"] = randn(Float32, h)
        out["$(pre).layer_scale2.lambda1"] = randn(Float32, h)
    end
    out["layernorm.weight"] = randn(Float32, h)
    out["layernorm.bias"] = randn(Float32, h)
    return out
end

@testset "DINOv2 architecture" begin
    cfg = _tiny_dinov2_config()
    m = Dinov2Model(cfg)
    @test length(m.layers) == cfg.num_hidden_layers
    for l in m.layers
        @test l.self_attn.causal == false
        @test l.self_attn.rope === nothing
        @test l.mlp.approx == false                 # exact GELU
        @test length(l.ls1) == cfg.hidden_size      # LayerScale vectors
        @test length(l.ls2) == cfg.hidden_size
    end
    @test_throws ArgumentError Dinov2Model(Dinov2Config(; use_swiglu_ffn=true))
end

@testset "DINOv2 forward shape (with and without registers)" begin
    Random.seed!(0xA1)
    np = (8 ÷ 4)^2
    for nreg in (0, 2)
        cfg = _tiny_dinov2_config(; num_register_tokens=nreg)
        m = Dinov2Model(cfg)
        x = randn(Float32, cfg.num_channels, cfg.image_size, cfg.image_size, 2)
        out = m(x)
        @test size(out) == (cfg.hidden_size, 1 + nreg + np, 2)
    end
end

@testset "LayerScale gates each residual branch" begin
    Random.seed!(0xA2)
    cfg = _tiny_dinov2_config(; num_register_tokens=0)
    m = Dinov2Model(cfg)
    l = m.layers[1]
    fill!(l.ls1, 0.0f0)
    fill!(l.ls2, 0.0f0)
    x = randn(Float32, cfg.hidden_size, 5, 2)
    @test l(x) ≈ x                                  # both branches zeroed → identity
end

@testset "registers inserted after CLS with no position embedding" begin
    Random.seed!(0xA3)
    cfg = _tiny_dinov2_config(; num_register_tokens=2)
    m = Dinov2Model(cfg)
    sd = _dinov2_synthetic_sd(cfg)
    load_state_dict!(m, sd)

    x = randn(Float32, cfg.num_channels, cfg.image_size, cfg.image_size, 2)
    emb = m.embeddings(x)
    nreg = cfg.num_register_tokens
    N = size(x, 4)

    # CLS slot = cls_token + position 1; register slots = register tokens verbatim.
    cls_expected = m.embeddings.cls_token .+ m.embeddings.position_embeddings[:, 1]
    @test all(emb[:, 1, j] ≈ cls_expected for j in 1:N)
    reg = repeat(reshape(m.embeddings.register_tokens, cfg.hidden_size, nreg, 1), 1, 1, N)
    @test emb[:, 2:(1 + nreg), :] ≈ reg
    # Patch slots = patch embeddings + positions 2:end.
    patches = m.embeddings.patch_embeddings(x)
    pos = reshape(m.embeddings.position_embeddings[:, 2:end], cfg.hidden_size, :, 1)
    @test emb[:, (2 + nreg):end, :] ≈ patches .+ pos
end

@testset "load_state_dict! populates DINOv2" begin
    Random.seed!(0xA4)
    cfg = _tiny_dinov2_config(; num_register_tokens=2)
    m = Dinov2Model(cfg)
    sd = _dinov2_synthetic_sd(cfg)
    load_state_dict!(m, sd)

    emb = m.embeddings
    @test emb.patch_embeddings.projection.weight ==
        reshape(sd["embeddings.patch_embeddings.projection.weight"], cfg.hidden_size, :)
    @test emb.cls_token == vec(sd["embeddings.cls_token"])
    @test emb.position_embeddings ==
        permutedims(dropdims(sd["embeddings.position_embeddings"]; dims=1), (2, 1))
    @test emb.register_tokens ==
        permutedims(dropdims(sd["embeddings.register_tokens"]; dims=1), (2, 1))

    l = m.layers[1]
    @test l.ls1 == sd["encoder.layer.0.layer_scale1.lambda1"]
    @test l.ls2 == sd["encoder.layer.0.layer_scale2.lambda1"]
    @test l.self_attn.wq.weight == sd["encoder.layer.0.attention.attention.query.weight"]
    @test l.self_attn.wo.bias == sd["encoder.layer.0.attention.output.dense.bias"]
    @test l.norm1.weight == sd["encoder.layer.0.norm1.weight"]
    @test l.mlp.c_fc.weight == sd["encoder.layer.0.mlp.fc1.weight"]
    @test m.layers[2].mlp.c_proj.weight == sd["encoder.layer.1.mlp.fc2.weight"]
    @test m.layernorm.bias == sd["layernorm.bias"]
end

@testset "dinov2_state_dict_map omits reshaped tensors" begin
    cfg = _tiny_dinov2_config()
    mp = dinov2_state_dict_map(cfg)
    @test haskey(mp, "encoder.layer.0.attention.attention.query.weight")
    @test haskey(mp, "encoder.layer.0.layer_scale1.lambda1")   # LayerScale via map
    @test haskey(mp, "layernorm.weight")
    @test !haskey(mp, "embeddings.patch_embeddings.projection.weight")
    @test !haskey(mp, "embeddings.cls_token")
    @test !haskey(mp, "embeddings.position_embeddings")
    @test !haskey(mp, "embeddings.register_tokens")
end
