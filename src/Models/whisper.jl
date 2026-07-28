"""
    WhisperConfig

Whisper ASR (`openai/whisper-base`). An encoder-decoder: a Conv1d audio frontend
+ pre-norm encoder over log-mel features, and a pre-norm decoder with causal
self-attention, cross-attention to the encoder, and a tied output head. Whisper's
`k_proj` carries no bias (its zero-initialized `wk.bias` is simply never loaded),
and `scale_embedding=false` so embeddings are unscaled.
"""
Base.@kwdef struct WhisperConfig
    vocab_size::Int = 51865
    num_mel_bins::Int = 80
    d_model::Int = 512
    encoder_layers::Int = 6
    encoder_attention_heads::Int = 8
    encoder_ffn_dim::Int = 2048
    decoder_layers::Int = 6
    decoder_attention_heads::Int = 8
    decoder_ffn_dim::Int = 2048
    max_source_positions::Int = 1500
    max_target_positions::Int = 448
    decoder_start_token_id::Int = 50258
    eos_token_id::Int = 50257
end

"""
    Conv1d{W, B}

1-D convolution matching PyTorch `Conv1d` (cross-correlation). `weight` is
`(out_channels, in_channels, kernel)`. Implemented by im2col + matmul so the
layout is explicit and testable; input/output are `(channels, length, batch)`.
"""
struct Conv1d{W,B}
    weight::W
    bias::B
    stride::Int
    pad::Int
end

function Conv1d(
    cin::Integer,
    cout::Integer,
    k::Integer;
    stride::Integer=1,
    pad::Integer=0,
    bias::Bool=true,
)
    return Conv1d(
        zeros(Float32, cout, cin, k), bias ? zeros(Float32, cout) : nothing, stride, pad
    )
end

function (c::Conv1d)(x::AbstractArray{<:Real,3})
    cin, len, n = size(x)
    cout, _, k = size(c.weight)
    lp = len + 2 * c.pad
    xp = if c.pad == 0
        x
    else
        buf = zeros(eltype(x), cin, lp, n)
        buf[:, (c.pad + 1):(c.pad + len), :] .= x
        buf
    end
    lout = (lp - k) ÷ c.stride + 1
    # im2col: each output position gathers a k-wide window, flattened
    # channel-fastest to match `reshape(weight, cout, cin*k)`.
    cols = Array{eltype(x)}(undef, cin * k, lout, n)
    @inbounds for o in 1:lout
        s = (o - 1) * c.stride
        cols[:, o, :] = reshape(view(xp, :, (s + 1):(s + k), :), cin * k, n)
    end
    wm = reshape(c.weight, cout, cin * k)
    out = reshape(wm * reshape(cols, cin * k, lout * n), cout, lout, n)
    return c.bias === nothing ? out : out .+ c.bias
end

Flux.@layer Conv1d

"""
    WhisperCrossAttention{Q, K, V, O}

Encoder-decoder attention: queries from the decoder, keys/values from the
encoder output. Reuses [`sdpa`](@ref). `wk` has no bias (Whisper convention).
"""
struct WhisperCrossAttention{Q,K,V,O}
    wq::Q
    wk::K
    wv::V
    wo::O
    num_heads::Int
    head_dim::Int
end

function (a::WhisperCrossAttention)(x_q::AbstractArray, x_kv::AbstractArray)
    d, sq, n = size(x_q)
    skv = size(x_kv, 2)
    H, hd = a.num_heads, a.head_dim

    q = reshape(a.wq(reshape(x_q, d, :)), hd, H, sq, n)
    k = reshape(a.wk(reshape(x_kv, d, :)), hd, H, skv, n)
    v = reshape(a.wv(reshape(x_kv, d, :)), hd, H, skv, n)

    # Into sdpa's (seq, head_dim, heads*batch) layout.
    q_flat = reshape(permutedims(q, (3, 1, 2, 4)), sq, hd, H * n)
    k_t = reshape(permutedims(k, (1, 3, 2, 4)), hd, skv, H * n)
    v_flat = reshape(permutedims(v, (3, 1, 2, 4)), skv, hd, H * n)

    out = sdpa(q_flat, k_t, v_flat; scale=sqrt(Float32(hd)))   # (sq, hd, H*n)
    out = permutedims(reshape(out, sq, hd, H, n), (2, 3, 1, 4)) # (hd, H, sq, n)
    return reshape(a.wo(reshape(out, d, sq * n)), d, sq, n)
end

Flux.@layer WhisperCrossAttention

struct WhisperEncoderLayer{A,N1,M,N2}
    self_attn::A
    self_attn_layer_norm::N1
    mlp::M
    final_layer_norm::N2
end

function (l::WhisperEncoderLayer)(x::AbstractArray)
    x = x .+ l.self_attn(l.self_attn_layer_norm(x))
    return x .+ l.mlp(l.final_layer_norm(x))
end

Flux.@layer WhisperEncoderLayer

"""
    WhisperEncoder{C1, C2, P, L, N}

Conv1d frontend (`conv1`, stride-1; `conv2`, stride-2) with GELU, plus the
sinusoidal position table, a pre-norm encoder stack, and a final LayerNorm.
Input is log-mel features `(num_mel_bins, frames, batch)`.
"""
struct WhisperEncoder{C1,C2,P,L,N}
    conv1::C1
    conv2::C2
    embed_positions::P
    layers::L
    layer_norm::N
end

function (e::WhisperEncoder)(features::AbstractArray{<:Real,3})
    h = _gelu_exact.(e.conv1(features))
    h = _gelu_exact.(e.conv2(h))                    # (d, frames/2, n)
    len = size(h, 2)
    h = h .+ reshape(e.embed_positions[:, 1:len], size(h, 1), len, 1)
    for layer in e.layers
        h = layer(h)
    end
    return e.layer_norm(h)
end

Flux.@layer WhisperEncoder

struct WhisperDecoderLayer{A,N1,C,N2,M,N3}
    self_attn::A
    self_attn_layer_norm::N1
    cross_attn::C
    encoder_attn_layer_norm::N2
    mlp::M
    final_layer_norm::N3
end

function (l::WhisperDecoderLayer)(x::AbstractArray, enc::AbstractArray)
    x = x .+ l.self_attn(l.self_attn_layer_norm(x))                 # causal
    x = x .+ l.cross_attn(l.encoder_attn_layer_norm(x), enc)        # cross
    return x .+ l.mlp(l.final_layer_norm(x))
end

Flux.@layer WhisperDecoderLayer

"""
    WhisperDecoder{E, P, L, N}

Token + learned position embeddings, a pre-norm stack of self+cross-attention
layers, and a final LayerNorm.
"""
struct WhisperDecoder{E,P,L,N}
    embed_tokens::E
    embed_positions::P
    layers::L
    layer_norm::N
end

function (dec::WhisperDecoder)(input_ids::AbstractMatrix{<:Integer}, enc::AbstractArray)
    seq = size(input_ids, 1)
    h = dec.embed_tokens(input_ids)
    h = h .+ reshape(dec.embed_positions[:, 1:seq], size(h, 1), seq, 1)
    for layer in dec.layers
        h = layer(h, enc)
    end
    return dec.layer_norm(h)
end

Flux.@layer WhisperDecoder

"""
    WhisperModel{C, E, D}

Full Whisper. `(model)(features, decoder_input_ids)` runs the encoder then the
decoder and projects to vocabulary logits `(vocab, seq, batch)` through the
tied token embedding.
"""
struct WhisperModel{C,E,D}
    config::C
    encoder::E
    decoder::D
end

function (m::WhisperModel)(
    features::AbstractArray{<:Real,3}, decoder_input_ids::AbstractMatrix{<:Integer}
)
    enc = m.encoder(features)
    dec = m.decoder(decoder_input_ids, enc)            # (d, seq, n)
    d, seq, n = size(dec)
    logits = permutedims(m.decoder.embed_tokens.weight) * reshape(dec, d, seq * n)
    return reshape(logits, :, seq, n)                  # (vocab, seq, n)
end

Flux.@layer WhisperModel

function WhisperModel(cfg::WhisperConfig)
    d = cfg.d_model
    e_hd = d ÷ cfg.encoder_attention_heads
    d_hd = d ÷ cfg.decoder_attention_heads

    encoder = WhisperEncoder(
        Conv1d(cfg.num_mel_bins, d, 3; stride=1, pad=1, bias=true),
        Conv1d(d, d, 3; stride=2, pad=1, bias=true),
        zeros(Float32, d, cfg.max_source_positions),
        [
            WhisperEncoderLayer(
                GQA(
                    d,
                    cfg.encoder_attention_heads,
                    cfg.encoder_attention_heads,
                    e_hd,
                    nothing;
                    qkv_bias=true,
                    wo_bias=true,
                    causal=false,
                ),
                LayerNorm(d),
                GeluMLP(d, cfg.encoder_ffn_dim; approx=false),
                LayerNorm(d),
            ) for _ in 1:(cfg.encoder_layers)
        ],
        LayerNorm(d),
    )

    decoder = WhisperDecoder(
        TokenEmbedding(cfg.vocab_size, d),
        zeros(Float32, d, cfg.max_target_positions),
        [
            WhisperDecoderLayer(
                GQA(
                    d,
                    cfg.decoder_attention_heads,
                    cfg.decoder_attention_heads,
                    d_hd,
                    nothing;
                    qkv_bias=true,
                    wo_bias=true,
                    causal=true,
                ),
                LayerNorm(d),
                WhisperCrossAttention(
                    Linear(d, d; bias=true),
                    Linear(d, d; bias=false),      # Whisper k_proj has no bias
                    Linear(d, d; bias=true),
                    Linear(d, d; bias=true),
                    cfg.decoder_attention_heads,
                    d_hd,
                ),
                LayerNorm(d),
                GeluMLP(d, cfg.decoder_ffn_dim; approx=false),
                LayerNorm(d),
            ) for _ in 1:(cfg.decoder_layers)
        ],
        LayerNorm(d),
    )

    return WhisperModel(cfg, encoder, decoder)
end

# Encoder/decoder attention layer entries (q/v/out biased; k weight only).
function _whisper_attn_map!(out, prefix::AbstractString, path::Tuple)
    out["$(prefix).q_proj.weight"] = ((path..., :wq, :weight), :as_is)
    out["$(prefix).q_proj.bias"] = ((path..., :wq, :bias), :as_is)
    out["$(prefix).k_proj.weight"] = ((path..., :wk, :weight), :as_is)   # no bias
    out["$(prefix).v_proj.weight"] = ((path..., :wv, :weight), :as_is)
    out["$(prefix).v_proj.bias"] = ((path..., :wv, :bias), :as_is)
    out["$(prefix).out_proj.weight"] = ((path..., :wo, :weight), :as_is)
    out["$(prefix).out_proj.bias"] = ((path..., :wo, :bias), :as_is)
    return out
end

_ln_map!(out, prefix, path) = begin
    out["$(prefix).weight"] = ((path..., :weight), :as_is)
    out["$(prefix).bias"] = ((path..., :bias), :as_is)
    out
end

"""
    whisper_state_dict_map(cfg::WhisperConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HF → internal table. Conv weights load `as_is` (3-D); embedding and position
tables load with `:transpose`. `k_proj.bias` and the tied `proj_out.weight` are
intentionally absent (the former does not exist; the head reuses the token
embedding).
"""
function whisper_state_dict_map(cfg::WhisperConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["model.encoder.conv1.weight"] = ((:encoder, :conv1, :weight), :as_is)
    out["model.encoder.conv1.bias"] = ((:encoder, :conv1, :bias), :as_is)
    out["model.encoder.conv2.weight"] = ((:encoder, :conv2, :weight), :as_is)
    out["model.encoder.conv2.bias"] = ((:encoder, :conv2, :bias), :as_is)
    out["model.encoder.embed_positions.weight"] = ((:encoder, :embed_positions), :transpose)
    for i in 0:(cfg.encoder_layers - 1)
        lp = (:encoder, :layers, i + 1)
        pre = "model.encoder.layers.$(i)"
        _whisper_attn_map!(out, "$(pre).self_attn", (lp..., :self_attn))
        _ln_map!(out, "$(pre).self_attn_layer_norm", (lp..., :self_attn_layer_norm))
        out["$(pre).fc1.weight"] = ((lp..., :mlp, :c_fc, :weight), :as_is)
        out["$(pre).fc1.bias"] = ((lp..., :mlp, :c_fc, :bias), :as_is)
        out["$(pre).fc2.weight"] = ((lp..., :mlp, :c_proj, :weight), :as_is)
        out["$(pre).fc2.bias"] = ((lp..., :mlp, :c_proj, :bias), :as_is)
        _ln_map!(out, "$(pre).final_layer_norm", (lp..., :final_layer_norm))
    end
    _ln_map!(out, "model.encoder.layer_norm", (:encoder, :layer_norm))

    out["model.decoder.embed_tokens.weight"] = (
        (:decoder, :embed_tokens, :weight), :transpose
    )
    out["model.decoder.embed_positions.weight"] = ((:decoder, :embed_positions), :transpose)
    for i in 0:(cfg.decoder_layers - 1)
        lp = (:decoder, :layers, i + 1)
        pre = "model.decoder.layers.$(i)"
        _whisper_attn_map!(out, "$(pre).self_attn", (lp..., :self_attn))
        _ln_map!(out, "$(pre).self_attn_layer_norm", (lp..., :self_attn_layer_norm))
        _whisper_attn_map!(out, "$(pre).encoder_attn", (lp..., :cross_attn))
        _ln_map!(out, "$(pre).encoder_attn_layer_norm", (lp..., :encoder_attn_layer_norm))
        out["$(pre).fc1.weight"] = ((lp..., :mlp, :c_fc, :weight), :as_is)
        out["$(pre).fc1.bias"] = ((lp..., :mlp, :c_fc, :bias), :as_is)
        out["$(pre).fc2.weight"] = ((lp..., :mlp, :c_proj, :weight), :as_is)
        out["$(pre).fc2.bias"] = ((lp..., :mlp, :c_proj, :bias), :as_is)
        _ln_map!(out, "$(pre).final_layer_norm", (lp..., :final_layer_norm))
    end
    _ln_map!(out, "model.decoder.layer_norm", (:decoder, :layer_norm))

    return out
end

"""
    load_state_dict!(m::WhisperModel, weights) -> m

Load a Whisper checkpoint via [`whisper_state_dict_map`](@ref). No manual
reshaping is needed: conv weights are 3-D `as_is`, the token/position tables
transpose, `k_proj` has no bias to load, and `proj_out` is tied to the token
embedding.
"""
function load_state_dict!(m::WhisperModel, weights::AbstractDict{String,<:AbstractArray})
    load_into!(m, weights, whisper_state_dict_map(m.config))
    return m
end

"""
    transcribe(m::WhisperModel, features, prompt_ids; max_new_tokens=64) -> Vector{Int}

Greedy decode from `prompt_ids` (the SOT / language / task prompt) conditioned on
`features` (log-mel, batch-1). Correctness-first: the decoder is recomputed each
step (no KV cache), so cost is quadratic in the output length.
"""
function transcribe(
    m::WhisperModel,
    features::AbstractArray{<:Real,3},
    prompt_ids::AbstractVector{<:Integer};
    max_new_tokens::Integer=64,
)
    enc = m.encoder(features)
    tokens = Int[Int(t) for t in prompt_ids]
    W = permutedims(m.decoder.embed_tokens.weight)
    for _ in 1:max_new_tokens
        dec = m.decoder(reshape(tokens, :, 1), enc)
        next = argmax(W * dec[:, end, 1]) - 1
        push!(tokens, next)
        next == m.config.eos_token_id && break
    end
    return tokens
end
