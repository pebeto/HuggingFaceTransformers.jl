"""
    ViTConfig

Vision Transformer (`google/vit-base-patch16-224` family). A pre-norm encoder
over image patches: a patch projection replaces token embeddings, a learned
`[CLS]` token and learned absolute position embeddings are added, and the
classifier reads the `[CLS]` position after a final LayerNorm. Field names
follow `config.json`.
"""
Base.@kwdef struct ViTConfig
    hidden_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    intermediate_size::Int
    image_size::Int = 224
    patch_size::Int = 16
    num_channels::Int = 3
    layer_norm_eps::Float64 = 1.0e-12
    num_labels::Int = 1000
    qkv_bias::Bool = true
    hf_prefix::String = "vit"
end

_vit_head_dim(cfg::ViTConfig) = cfg.hidden_size ÷ cfg.num_attention_heads
_vit_num_patches(cfg::ViTConfig) = (cfg.image_size ÷ cfg.patch_size)^2

"""
    ViTPatchEmbeddings{P}

Non-overlapping patch projection. HF uses a `Conv2d` with kernel = stride =
`patch_size`, which for non-overlapping patches equals flattening each patch and
applying a `Linear`. `projection.weight` is the conv weight reshaped to
`(hidden, num_channels * patch_size^2)`.
"""
struct ViTPatchEmbeddings{P}
    projection::P
    patch_size::Int
    num_channels::Int
end

# pixel_values: (channels, height, width, batch). Patches are flattened
# channel-fastest to match `reshape(conv_weight, hidden, :)`, and enumerated
# with the width-patch index fastest to match HF's `flatten(2)` order.
function (e::ViTPatchEmbeddings)(pixel_values::AbstractArray{<:Real,4})
    c, h, w, n = size(pixel_values)
    p = e.patch_size
    gh, gw = h ÷ p, w ÷ p
    xr = reshape(pixel_values, c, p, gh, p, gw, n)     # (c, i, ih, j, iw, n)
    perm = permutedims(xr, (1, 2, 4, 5, 3, 6))         # (c, i, j, iw, ih, n)
    patches = reshape(perm, c * p * p, gw * gh, n)      # (feat, patch, n)
    flat = e.projection(reshape(patches, c * p * p, :)) # (hidden, patch*n)
    return reshape(flat, :, gw * gh, n)
end

Flux.@layer ViTPatchEmbeddings

"""
    ViTEmbeddings{PE, C, P}

Patch projection + prepended `[CLS]` token + learned position embeddings.
Returns `(hidden, num_patches + 1, batch)`.
"""
struct ViTEmbeddings{PE,C,P}
    patch_embeddings::PE
    cls_token::C
    position_embeddings::P
end

function (e::ViTEmbeddings)(pixel_values::AbstractArray{<:Real,4})
    patches = e.patch_embeddings(pixel_values)          # (hidden, np, n)
    n = size(patches, 3)
    cls = repeat(reshape(e.cls_token, :, 1, 1), 1, 1, n)
    h = cat(cls, patches; dims=2)                       # (hidden, np+1, n)
    pos = reshape(e.position_embeddings, size(e.position_embeddings)..., 1)
    return h .+ pos
end

Flux.@layer ViTEmbeddings

"""
    ViTModel{E, L, N}

Encoder trunk: embeddings → pre-norm transformer stack → final LayerNorm.
Returns per-token hidden states `(hidden, num_patches + 1, batch)`.
"""
struct ViTModel{E,L,N}
    embeddings::E
    layers::L
    layernorm::N
end

function (m::ViTModel)(pixel_values::AbstractArray{<:Real,4})
    h = m.embeddings(pixel_values)
    for layer in m.layers
        h = layer(h)
    end
    return m.layernorm(h)
end

Flux.@layer ViTModel

"""
    ViTForImageClassification{C, M, H}

`ViTModel` trunk plus a linear classifier on the `[CLS]` token (position 1)
after the final LayerNorm. Forward returns logits `(num_labels, batch)`.
"""
struct ViTForImageClassification{C,M,H}
    config::C
    vit::M
    classifier::H
end

function (lm::ViTForImageClassification)(pixel_values::AbstractArray{<:Real,4})
    h = lm.vit(pixel_values)          # (hidden, np+1, n)
    cls = h[:, 1, :]                  # (hidden, n)
    return lm.classifier(cls)         # (num_labels, n)
end

Flux.@layer ViTForImageClassification

"""
    ViTForImageClassification(cfg::ViTConfig)

Materialize a fresh, randomly-initialized ViT classifier. Attention is
bidirectional MHA with no rotary; the MLP is exact-GELU; the block is pre-norm.
"""
function ViTForImageClassification(cfg::ViTConfig)
    eps = Float32(cfg.layer_norm_eps)
    head_dim = _vit_head_dim(cfg)
    patch_dim = cfg.num_channels * cfg.patch_size^2
    num_positions = _vit_num_patches(cfg) + 1

    embeddings = ViTEmbeddings(
        ViTPatchEmbeddings(
            Linear(patch_dim, cfg.hidden_size; bias=true), cfg.patch_size, cfg.num_channels
        ),
        zeros(Float32, cfg.hidden_size),
        zeros(Float32, cfg.hidden_size, num_positions),
    )

    layers = [
        DecoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_attention_heads,    # MHA
                head_dim,
                nothing;                    # no RoPE
                qkv_bias=cfg.qkv_bias,
                wo_bias=true,
                causal=false,               # bidirectional
            ),
            LayerNorm(cfg.hidden_size, eps),     # layernorm_before
            LayerNorm(cfg.hidden_size, eps),     # layernorm_after
            GeluMLP(cfg.hidden_size, cfg.intermediate_size; approx=false),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    vit = ViTModel(embeddings, layers, LayerNorm(cfg.hidden_size, eps))
    classifier = Linear(cfg.hidden_size, cfg.num_labels; bias=true)
    return ViTForImageClassification(cfg, vit, classifier)
end

"""
    vit_state_dict_map(cfg::ViTConfig) -> Dict{String, Tuple{Tuple, Symbol}}

HuggingFace → internal table for the encoder, final LayerNorm, and classifier.
The patch conv, `[CLS]` token, and position embeddings need reshaping, so
`load_state_dict!` handles those by hand.
"""
function vit_state_dict_map(cfg::ViTConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()
    p = cfg.hf_prefix

    for i in 0:(cfg.num_hidden_layers - 1)
        lp = (:vit, :layers, i + 1)
        pre = "$(p).encoder.layer.$(i)"
        out["$(pre).attention.attention.query.weight"] = ((lp..., :self_attn, :wq, :weight), :as_is)
        out["$(pre).attention.attention.query.bias"] = ((lp..., :self_attn, :wq, :bias), :as_is)
        out["$(pre).attention.attention.key.weight"] = ((lp..., :self_attn, :wk, :weight), :as_is)
        out["$(pre).attention.attention.key.bias"] = ((lp..., :self_attn, :wk, :bias), :as_is)
        out["$(pre).attention.attention.value.weight"] = ((lp..., :self_attn, :wv, :weight), :as_is)
        out["$(pre).attention.attention.value.bias"] = ((lp..., :self_attn, :wv, :bias), :as_is)
        out["$(pre).attention.output.dense.weight"] = ((lp..., :self_attn, :wo, :weight), :as_is)
        out["$(pre).attention.output.dense.bias"] = ((lp..., :self_attn, :wo, :bias), :as_is)
        out["$(pre).intermediate.dense.weight"] = ((lp..., :mlp, :c_fc, :weight), :as_is)
        out["$(pre).intermediate.dense.bias"] = ((lp..., :mlp, :c_fc, :bias), :as_is)
        out["$(pre).output.dense.weight"] = ((lp..., :mlp, :c_proj, :weight), :as_is)
        out["$(pre).output.dense.bias"] = ((lp..., :mlp, :c_proj, :bias), :as_is)
        out["$(pre).layernorm_before.weight"] = ((lp..., :input_layernorm, :weight), :as_is)
        out["$(pre).layernorm_before.bias"] = ((lp..., :input_layernorm, :bias), :as_is)
        out["$(pre).layernorm_after.weight"] = ((lp..., :post_attention_layernorm, :weight), :as_is)
        out["$(pre).layernorm_after.bias"] = ((lp..., :post_attention_layernorm, :bias), :as_is)
    end

    out["$(p).layernorm.weight"] = ((:vit, :layernorm, :weight), :as_is)
    out["$(p).layernorm.bias"] = ((:vit, :layernorm, :bias), :as_is)
    out["classifier.weight"] = ((:classifier, :weight), :as_is)
    out["classifier.bias"] = ((:classifier, :bias), :as_is)
    return out
end

"""
    load_state_dict!(lm::ViTForImageClassification, weights) -> lm

Load a ViT classifier. The encoder/head go through [`vit_state_dict_map`](@ref);
the patch conv weight `(hidden, C, P, P)` is reshaped to the `Linear`
`(hidden, C*P*P)`, and the `(1,1,H)` / `(1,S,H)` `[CLS]` and position tensors
are squeezed and transposed by hand.
"""
function load_state_dict!(
    lm::ViTForImageClassification, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(lm, weights, vit_state_dict_map(lm.config))

    p = lm.config.hf_prefix
    emb = lm.vit.embeddings

    conv = weights["$(p).embeddings.patch_embeddings.projection.weight"]  # (hidden, C, P, P)
    copyto!(emb.patch_embeddings.projection.weight, reshape(Array(conv), lm.config.hidden_size, :))
    copyto!(
        emb.patch_embeddings.projection.bias,
        weights["$(p).embeddings.patch_embeddings.projection.bias"],
    )

    copyto!(emb.cls_token, vec(weights["$(p).embeddings.cls_token"]))    # (1,1,H) → (H,)
    pos = weights["$(p).embeddings.position_embeddings"]                  # (1, S, H)
    copyto!(emb.position_embeddings, permutedims(dropdims(pos; dims=1), (2, 1)))  # (H, S)
    return lm
end
