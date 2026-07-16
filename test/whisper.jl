using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: whisper_state_dict_map, load_state_dict!, Conv1d, WhisperCrossAttention

function _tiny_whisper_config()
    return WhisperConfig(;
        vocab_size=20,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_source_positions=6,
        max_target_positions=8,
        decoder_start_token_id=1,
        eos_token_id=19,
    )
end

function _whisper_attn_sd!(out, pre, h)
    out["$(pre).q_proj.weight"] = randn(Float32, h, h)
    out["$(pre).q_proj.bias"] = randn(Float32, h)
    out["$(pre).k_proj.weight"] = randn(Float32, h, h)   # no bias
    out["$(pre).v_proj.weight"] = randn(Float32, h, h)
    out["$(pre).v_proj.bias"] = randn(Float32, h)
    out["$(pre).out_proj.weight"] = randn(Float32, h, h)
    out["$(pre).out_proj.bias"] = randn(Float32, h)
end

_ln_sd!(out, pre, h) = (out["$(pre).weight"] = randn(Float32, h); out["$(pre).bias"] = randn(Float32, h))

function _whisper_synthetic_sd(cfg::WhisperConfig)
    d = cfg.d_model
    out = Dict{String,Array{Float32}}()
    out["model.encoder.conv1.weight"] = randn(Float32, d, cfg.num_mel_bins, 3)
    out["model.encoder.conv1.bias"] = randn(Float32, d)
    out["model.encoder.conv2.weight"] = randn(Float32, d, d, 3)
    out["model.encoder.conv2.bias"] = randn(Float32, d)
    out["model.encoder.embed_positions.weight"] = randn(Float32, cfg.max_source_positions, d)
    for i in 0:(cfg.encoder_layers - 1)
        pre = "model.encoder.layers.$(i)"
        _whisper_attn_sd!(out, "$(pre).self_attn", d)
        _ln_sd!(out, "$(pre).self_attn_layer_norm", d)
        out["$(pre).fc1.weight"] = randn(Float32, cfg.encoder_ffn_dim, d)
        out["$(pre).fc1.bias"] = randn(Float32, cfg.encoder_ffn_dim)
        out["$(pre).fc2.weight"] = randn(Float32, d, cfg.encoder_ffn_dim)
        out["$(pre).fc2.bias"] = randn(Float32, d)
        _ln_sd!(out, "$(pre).final_layer_norm", d)
    end
    _ln_sd!(out, "model.encoder.layer_norm", d)

    out["model.decoder.embed_tokens.weight"] = randn(Float32, cfg.vocab_size, d)
    out["model.decoder.embed_positions.weight"] = randn(Float32, cfg.max_target_positions, d)
    for i in 0:(cfg.decoder_layers - 1)
        pre = "model.decoder.layers.$(i)"
        _whisper_attn_sd!(out, "$(pre).self_attn", d)
        _ln_sd!(out, "$(pre).self_attn_layer_norm", d)
        _whisper_attn_sd!(out, "$(pre).encoder_attn", d)
        _ln_sd!(out, "$(pre).encoder_attn_layer_norm", d)
        out["$(pre).fc1.weight"] = randn(Float32, cfg.decoder_ffn_dim, d)
        out["$(pre).fc1.bias"] = randn(Float32, cfg.decoder_ffn_dim)
        out["$(pre).fc2.weight"] = randn(Float32, d, cfg.decoder_ffn_dim)
        out["$(pre).fc2.bias"] = randn(Float32, d)
        _ln_sd!(out, "$(pre).final_layer_norm", d)
    end
    _ln_sd!(out, "model.decoder.layer_norm", d)
    return out
end

function _ref_conv1d(W, b, x, stride, pad)
    cout, cin, k = size(W)
    _, len, n = size(x)
    lp = len + 2pad
    xp = zeros(Float32, cin, lp, n)
    xp[:, (pad + 1):(pad + len), :] = x
    lout = (lp - k) ÷ stride + 1
    out = zeros(Float32, cout, lout, n)
    for nn in 1:n, o in 1:lout, co in 1:cout
        s = (o - 1) * stride
        acc = b[co]
        for ci in 1:cin, kk in 1:k
            acc += W[co, ci, kk] * xp[ci, s + kk, nn]
        end
        out[co, o, nn] = acc
    end
    return out
end

function _ref_cross_attn(a, xq, xkv)
    d, sq, n = size(xq)
    skv = size(xkv, 2)
    H, hd = a.num_heads, a.head_dim
    q = reshape(a.wq(reshape(xq, d, :)), d, sq, n)
    k = reshape(a.wk(reshape(xkv, d, :)), d, skv, n)
    v = reshape(a.wv(reshape(xkv, d, :)), d, skv, n)
    out = zeros(Float32, d, sq, n)
    for nn in 1:n, h in 1:H
        r = ((h - 1) * hd + 1):(h * hd)
        for i in 1:sq
            scores = Float32[sum(q[r, i, nn] .* k[r, j, nn]) / sqrt(Float32(hd)) for j in 1:skv]
            w = softmax(scores)
            acc = zeros(Float32, hd)
            for j in 1:skv
                acc .+= w[j] .* v[r, j, nn]
            end
            out[r, i, nn] = acc
        end
    end
    return reshape(a.wo(reshape(out, d, :)), d, sq, n)
end

@testset "Conv1d matches explicit reference" begin
    Random.seed!(0xB1)
    c = Conv1d(4, 8, 3; stride=2, pad=1, bias=true)
    copyto!(c.weight, randn(Float32, 8, 4, 3))
    copyto!(c.bias, randn(Float32, 8))
    x = randn(Float32, 4, 6, 2)
    @test c(x) ≈ _ref_conv1d(c.weight, c.bias, x, c.stride, c.pad)
end

@testset "cross-attention matches explicit reference" begin
    Random.seed!(0xB2)
    a = WhisperCrossAttention(
        Allspark.Layers.Linear(8, 8; bias=true),
        Allspark.Layers.Linear(8, 8; bias=false),
        Allspark.Layers.Linear(8, 8; bias=true),
        Allspark.Layers.Linear(8, 8; bias=true),
        2, 4,
    )
    for l in (a.wq, a.wv, a.wo)
        copyto!(l.weight, randn(Float32, 8, 8))
        copyto!(l.bias, randn(Float32, 8))
    end
    copyto!(a.wk.weight, randn(Float32, 8, 8))
    xq = randn(Float32, 8, 3, 2)
    xkv = randn(Float32, 8, 5, 2)
    @test a(xq, xkv) ≈ _ref_cross_attn(a, xq, xkv)
end

@testset "Whisper architecture" begin
    cfg = _tiny_whisper_config()
    m = WhisperModel(cfg)
    @test length(m.encoder.layers) == cfg.encoder_layers
    @test length(m.decoder.layers) == cfg.decoder_layers
    @test all(l.self_attn.causal == false for l in m.encoder.layers)   # encoder bidirectional
    @test all(l.self_attn.causal == true for l in m.decoder.layers)    # decoder causal
    @test m.decoder.layers[1].cross_attn.wk.bias === nothing           # k_proj has no bias
end

@testset "Whisper forward shape" begin
    Random.seed!(0xB3)
    cfg = _tiny_whisper_config()
    m = WhisperModel(cfg)
    features = randn(Float32, cfg.num_mel_bins, 6, 2)
    ids = rand(0:(cfg.vocab_size - 1), 4, 2)
    logits = m(features, ids)
    @test size(logits) == (cfg.vocab_size, 4, 2)
end

@testset "decoder self-attention is causal" begin
    Random.seed!(0xB4)
    cfg = _tiny_whisper_config()
    m = WhisperModel(cfg)
    load_state_dict!(m, _whisper_synthetic_sd(cfg))
    features = randn(Float32, cfg.num_mel_bins, 6, 1)
    ids = reshape([2, 5, 9, 3], :, 1)

    l1 = m(features, ids)
    ids2 = copy(ids)
    ids2[end, 1] = 7                                # change only the last token
    l2 = m(features, ids2)
    @test l1[:, 1, 1] ≈ l2[:, 1, 1]                 # position 1 unaffected (causal)
    @test !(l1[:, end, 1] ≈ l2[:, end, 1])          # last position changes
end

@testset "load_state_dict! populates Whisper" begin
    Random.seed!(0xB5)
    cfg = _tiny_whisper_config()
    m = WhisperModel(cfg)
    sd = _whisper_synthetic_sd(cfg)
    load_state_dict!(m, sd)
    d = cfg.d_model

    # Conv weights load as-is (3-D); position/token tables transpose.
    @test m.encoder.conv1.weight == sd["model.encoder.conv1.weight"]
    @test m.encoder.conv2.bias == sd["model.encoder.conv2.bias"]
    @test m.encoder.embed_positions ==
        permutedims(sd["model.encoder.embed_positions.weight"], (2, 1))
    @test m.decoder.embed_tokens.weight ==
        permutedims(sd["model.decoder.embed_tokens.weight"], (2, 1))
    @test m.decoder.embed_positions ==
        permutedims(sd["model.decoder.embed_positions.weight"], (2, 1))

    # Encoder self-attn + decoder self/cross attn (k weight loaded, no k bias).
    el = m.encoder.layers[1]
    @test el.self_attn.wq.weight == sd["model.encoder.layers.0.self_attn.q_proj.weight"]
    @test el.mlp.c_fc.weight == sd["model.encoder.layers.0.fc1.weight"]
    @test el.final_layer_norm.bias == sd["model.encoder.layers.0.final_layer_norm.bias"]

    dl = m.decoder.layers[1]
    @test dl.self_attn.wq.weight == sd["model.decoder.layers.0.self_attn.q_proj.weight"]
    @test dl.cross_attn.wk.weight == sd["model.decoder.layers.0.encoder_attn.k_proj.weight"]
    @test dl.cross_attn.wv.bias == sd["model.decoder.layers.0.encoder_attn.v_proj.bias"]
    @test dl.encoder_attn_layer_norm.weight ==
        sd["model.decoder.layers.0.encoder_attn_layer_norm.weight"]
    @test m.decoder.layer_norm.bias == sd["model.decoder.layer_norm.bias"]
end

@testset "transcribe greedy-decodes from a prompt" begin
    Random.seed!(0xB6)
    cfg = _tiny_whisper_config()
    m = WhisperModel(cfg)
    load_state_dict!(m, _whisper_synthetic_sd(cfg))
    features = randn(Float32, cfg.num_mel_bins, 6, 1)

    out = transcribe(m, features, [cfg.decoder_start_token_id]; max_new_tokens=5)
    @test out isa Vector{Int}
    @test out[1] == cfg.decoder_start_token_id
    @test length(out) <= 1 + 5
    @test all(0 .<= out .< cfg.vocab_size)
end

@testset "whisper_state_dict_map omits k bias and tied head" begin
    cfg = _tiny_whisper_config()
    mp = whisper_state_dict_map(cfg)
    @test haskey(mp, "model.encoder.conv1.weight")
    @test mp["model.encoder.conv1.weight"][2] == :as_is
    @test mp["model.decoder.embed_tokens.weight"][2] == :transpose
    @test haskey(mp, "model.decoder.layers.0.self_attn.k_proj.weight")
    @test !haskey(mp, "model.decoder.layers.0.self_attn.k_proj.bias")   # no k bias
    @test !haskey(mp, "proj_out.weight")                                # tied
end
