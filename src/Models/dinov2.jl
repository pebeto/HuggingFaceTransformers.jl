"""
    Dinov2Config

DINOv2 image encoder (`facebook/dinov2-base`). A ViT-family pre-norm backbone
with two additions: LayerScale on each residual branch (element-wise multiply by
a learned `lambda1`) and optional register tokens inserted after the `[CLS]`
token. `layer_norm_eps = 1e-6`, exact GELU, `patch_size=14`, `image_size=518`.
The giant variant's SwiGLU FFN is not supported yet (`use_swiglu_ffn=true`
errors).
"""
Base.@kwdef struct Dinov2Config
    hidden_size::Int = 768
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 12
    intermediate_size::Int = 3072
    image_size::Int = 518
    patch_size::Int = 14
    num_channels::Int = 3
    layer_norm_eps::Float64 = 1.0e-6
    layerscale_value::Float64 = 1.0
    num_register_tokens::Int = 0
    qkv_bias::Bool = true
    use_swiglu_ffn::Bool = false
    hf_prefix::String = ""
end

_dinov2_num_patches(cfg::Dinov2Config) = (cfg.image_size ÷ cfg.patch_size)^2

"""
    Dinov2Embeddings{PE, C, R, P}

Patch projection + prepended `[CLS]` token + learned position embeddings. When
`num_register_tokens > 0`, register tokens are inserted after `[CLS]` and before
the patches, and they receive no position embedding (position embeddings are
added to `[cls, patches]` first).
"""
struct Dinov2Embeddings{PE,C,R,P}
    patch_embeddings::PE
    cls_token::C
    register_tokens::R
    position_embeddings::P
    num_register_tokens::Int
end

function (e::Dinov2Embeddings)(pixel_values::AbstractArray{<:Real,4})
    patches = e.patch_embeddings(pixel_values)         # (hidden, np, n)
    n = size(patches, 3)
    cls = repeat(reshape(e.cls_token, :, 1, 1), 1, 1, n)
    h = cat(cls, patches; dims=2)                      # (hidden, np+1, n)
    h = h .+ reshape(e.position_embeddings, size(e.position_embeddings)..., 1)

    if e.num_register_tokens > 0
        reg = repeat(
            reshape(e.register_tokens, size(e.register_tokens, 1), e.num_register_tokens, 1),
            1, 1, n,
        )
        h = cat(h[:, 1:1, :], reg, h[:, 2:end, :]; dims=2)   # [cls, regs, patches]
    end
    return h
end

Flux.@layer Dinov2Embeddings

"""
    Dinov2Layer{A, N1, N2, M}

Pre-norm block with LayerScale: `x = x + ls1 ⊙ attn(norm1(x))` then
`x = x + ls2 ⊙ mlp(norm2(x))`. `ls1`/`ls2` are the per-channel `lambda1` vectors.
"""
struct Dinov2Layer{A,N1,N2,M}
    self_attn::A
    norm1::N1
    norm2::N2
    mlp::M
    ls1::Vector{Float32}
    ls2::Vector{Float32}
end

function (l::Dinov2Layer)(x::AbstractArray)
    x = x .+ l.ls1 .* l.self_attn(l.norm1(x))
    return x .+ l.ls2 .* l.mlp(l.norm2(x))
end

Flux.@layer Dinov2Layer

"""
    Dinov2Model{E, L, N}

embeddings → pre-norm encoder (with LayerScale) → final LayerNorm. Returns the
per-token hidden states `(hidden, seq, batch)`; the pooled image feature is the
`[CLS]` token `output[:, 1, :]`.
"""
struct Dinov2Model{C,E,L,N}
    config::C
    embeddings::E
    layers::L
    layernorm::N
end

function (m::Dinov2Model)(pixel_values::AbstractArray{<:Real,4})
    h = m.embeddings(pixel_values)
    for layer in m.layers
        h = layer(h)
    end
    return m.layernorm(h)
end

Flux.@layer Dinov2Model

"""
    Dinov2Model(cfg::Dinov2Config)

Materialize a fresh DINOv2 backbone. Attention is bidirectional MHA (no rotary);
the MLP is exact GELU; each residual branch carries a LayerScale vector.
"""
function Dinov2Model(cfg::Dinov2Config)
    cfg.use_swiglu_ffn && throw(
        ArgumentError("DINOv2 SwiGLU FFN (giant variant) is not supported yet"),
    )
    eps = Float32(cfg.layer_norm_eps)
    head_dim = cfg.hidden_size ÷ cfg.num_attention_heads
    num_positions = _dinov2_num_patches(cfg) + 1
    patch_dim = cfg.num_channels * cfg.patch_size^2
    ls_init = Float32(cfg.layerscale_value)

    embeddings = Dinov2Embeddings(
        ViTPatchEmbeddings(
            Linear(patch_dim, cfg.hidden_size; bias=true), cfg.patch_size, cfg.num_channels
        ),
        zeros(Float32, cfg.hidden_size),
        zeros(Float32, cfg.hidden_size, cfg.num_register_tokens),
        zeros(Float32, cfg.hidden_size, num_positions),
        cfg.num_register_tokens,
    )

    layers = [
        Dinov2Layer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_attention_heads,        # MHA
                head_dim,
                nothing;                        # no rotary
                qkv_bias=cfg.qkv_bias,
                wo_bias=true,
                causal=false,
            ),
            LayerNorm(cfg.hidden_size, eps),
            LayerNorm(cfg.hidden_size, eps),
            GeluMLP(cfg.hidden_size, cfg.intermediate_size; approx=false),
            fill(ls_init, cfg.hidden_size),
            fill(ls_init, cfg.hidden_size),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    return Dinov2Model(cfg, embeddings, layers, LayerNorm(cfg.hidden_size, eps))
end

"""
    dinov2_state_dict_map(cfg::Dinov2Config) -> Dict{String, Tuple{Tuple, Symbol}}

HF → internal table for the encoder and final LayerNorm (including the LayerScale
`lambda1` vectors). The patch conv, `[CLS]`, register, and position tensors need
reshaping, so `load_state_dict!` handles those by hand.
"""
function dinov2_state_dict_map(cfg::Dinov2Config)
    out = Dict{String,Tuple{Tuple,Symbol}}()
    p = cfg.hf_prefix
    pfx(s) = isempty(p) ? s : "$(p).$(s)"

    out[pfx("embeddings.patch_embeddings.projection.bias")] = (
        (:embeddings, :patch_embeddings, :projection, :bias), :as_is
    )

    for i in 0:(cfg.num_hidden_layers - 1)
        lp = (:layers, i + 1)
        pre = pfx("encoder.layer.$(i)")
        out["$(pre).norm1.weight"] = ((lp..., :norm1, :weight), :as_is)
        out["$(pre).norm1.bias"] = ((lp..., :norm1, :bias), :as_is)
        out["$(pre).attention.attention.query.weight"] = ((lp..., :self_attn, :wq, :weight), :as_is)
        out["$(pre).attention.attention.query.bias"] = ((lp..., :self_attn, :wq, :bias), :as_is)
        out["$(pre).attention.attention.key.weight"] = ((lp..., :self_attn, :wk, :weight), :as_is)
        out["$(pre).attention.attention.key.bias"] = ((lp..., :self_attn, :wk, :bias), :as_is)
        out["$(pre).attention.attention.value.weight"] = ((lp..., :self_attn, :wv, :weight), :as_is)
        out["$(pre).attention.attention.value.bias"] = ((lp..., :self_attn, :wv, :bias), :as_is)
        out["$(pre).attention.output.dense.weight"] = ((lp..., :self_attn, :wo, :weight), :as_is)
        out["$(pre).attention.output.dense.bias"] = ((lp..., :self_attn, :wo, :bias), :as_is)
        out["$(pre).layer_scale1.lambda1"] = ((lp..., :ls1), :as_is)
        out["$(pre).norm2.weight"] = ((lp..., :norm2, :weight), :as_is)
        out["$(pre).norm2.bias"] = ((lp..., :norm2, :bias), :as_is)
        out["$(pre).mlp.fc1.weight"] = ((lp..., :mlp, :c_fc, :weight), :as_is)
        out["$(pre).mlp.fc1.bias"] = ((lp..., :mlp, :c_fc, :bias), :as_is)
        out["$(pre).mlp.fc2.weight"] = ((lp..., :mlp, :c_proj, :weight), :as_is)
        out["$(pre).mlp.fc2.bias"] = ((lp..., :mlp, :c_proj, :bias), :as_is)
        out["$(pre).layer_scale2.lambda1"] = ((lp..., :ls2), :as_is)
    end

    out[pfx("layernorm.weight")] = ((:layernorm, :weight), :as_is)
    out[pfx("layernorm.bias")] = ((:layernorm, :bias), :as_is)
    return out
end

"""
    load_state_dict!(m::Dinov2Model, weights) -> m

Load a DINOv2 backbone. Most parameters (including the LayerScale vectors) go
through [`dinov2_state_dict_map`](@ref); the patch conv is reshaped to its
`Linear`, and the `[CLS]`, register, and position tensors are squeezed and
transposed by hand. Any `mask_token` (training-only) is ignored.
"""
function load_state_dict!(m::Dinov2Model, weights::AbstractDict{String,<:AbstractArray})
    load_into!(m, weights, dinov2_state_dict_map(m.config))

    p = m.config.hf_prefix
    pfx(s) = isempty(p) ? s : "$(p).$(s)"
    emb = m.embeddings

    conv = weights[pfx("embeddings.patch_embeddings.projection.weight")]  # (hidden, C, P, P)
    copyto!(emb.patch_embeddings.projection.weight, reshape(Array(conv), m.config.hidden_size, :))
    copyto!(emb.cls_token, vec(weights[pfx("embeddings.cls_token")]))     # (1,1,H) → (H,)
    pos = weights[pfx("embeddings.position_embeddings")]                   # (1, S, H)
    copyto!(emb.position_embeddings, permutedims(dropdims(pos; dims=1), (2, 1)))

    if m.config.num_register_tokens > 0
        reg = weights[pfx("embeddings.register_tokens")]                   # (1, R, H)
        copyto!(emb.register_tokens, permutedims(dropdims(reg; dims=1), (2, 1)))
    end
    return m
end
