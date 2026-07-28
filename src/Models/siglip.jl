"""
    SiglipVisionConfig

Vision tower hyperparameters for SigLIP (`google/siglip-base-patch16-224`). A
ViT-style pre-norm encoder with `gelu_pytorch_tanh` MLPs at
`layer_norm_eps = 1e-6`. The tower carries no `[CLS]` token and pools its patch
tokens through a learned-probe attention head.
"""
Base.@kwdef struct SiglipVisionConfig
    hidden_size::Int = 768
    intermediate_size::Int = 3072
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 12
    num_channels::Int = 3
    image_size::Int = 224
    patch_size::Int = 16
    layer_norm_eps::Float64 = 1.0e-6
end

"""
    SiglipTextConfig

Text tower hyperparameters for SigLIP. The encoder has the same pre-norm shape
as the vision tower, but it pools the last token of the sequence and applies a
linear `head` to reach `projection_size`.
"""
Base.@kwdef struct SiglipTextConfig
    vocab_size::Int = 32000
    hidden_size::Int = 768
    intermediate_size::Int = 3072
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 12
    max_position_embeddings::Int = 64
    projection_size::Int = 768
    layer_norm_eps::Float64 = 1.0e-6
end

"""
    SiglipConfig

Pairs a [`SiglipVisionConfig`](@ref) with a [`SiglipTextConfig`](@ref).
[`SiglipModel`](@ref) L2-normalizes the embedding from each tower and scores
them against each other using `logit_scale` and `logit_bias`.
"""
Base.@kwdef struct SiglipConfig
    vision::SiglipVisionConfig = SiglipVisionConfig()
    text::SiglipTextConfig = SiglipTextConfig()
end

# A pre-norm encoder layer shared by both towers: bidirectional biased-QKV MHA
# (no rotary) + tanh-GELU MLP. DecoderLayer's forward is exactly SigLIP's
# `x = x + attn(ln1(x)); x = x + mlp(ln2(x))`.
function _siglip_layer(hidden::Integer, nheads::Integer, inter::Integer, eps::Float32)
    hd = hidden ÷ nheads
    return DecoderLayer(
        GQA(hidden, nheads, nheads, hd, nothing; qkv_bias=true, wo_bias=true, causal=false),
        LayerNorm(hidden, eps),
        LayerNorm(hidden, eps),
        GeluMLP(hidden, inter; approx=true),
    )
end

"""
    SiglipVisionEmbeddings{PE, P}

Patch projection (no `[CLS]` token) plus a learned position-embedding table
`(hidden, num_patches)` added to every patch.
"""
struct SiglipVisionEmbeddings{PE,P}
    patch_embeddings::PE
    position_embeddings::P
end

function (e::SiglipVisionEmbeddings)(pixel_values::AbstractArray{<:Real,4})
    patches = e.patch_embeddings(pixel_values)   # (hidden, np, n)
    pos = reshape(e.position_embeddings, size(e.position_embeddings)..., 1)
    return patches .+ pos
end

Flux.@layer SiglipVisionEmbeddings

"""
    SiglipAttentionPoolingHead{Q, K, V, O, N, M}

Pools patch tokens with a single learned probe query (HF's
`nn.MultiheadAttention`). `wq`/`wk`/`wv` are the split of the fused
`in_proj_weight`; `wo` is `out_proj`. Output is `probe_attn + mlp(layernorm(...))`.
"""
struct SiglipAttentionPoolingHead{Q,K,V,O,N,M}
    wq::Q
    wk::K
    wv::V
    wo::O
    probe::Vector{Float32}
    layernorm::N
    mlp::M
    num_heads::Int
    head_dim::Int
end

function (h::SiglipAttentionPoolingHead)(hs::AbstractArray)
    hidden, np, n = size(hs)
    H, hd = h.num_heads, h.head_dim

    q = reshape(h.wq(h.probe), hd, H)                 # (hd, H), one query
    hsf = reshape(hs, hidden, np * n)
    k = reshape(h.wk(hsf), hd, H, np, n)
    v = reshape(h.wv(hsf), hd, H, np, n)

    scores = dropdims(sum(reshape(q, hd, H, 1, 1) .* k; dims=1); dims=1)  # (H, np, n)
    scores = scores ./ sqrt(Float32(hd))
    weights = softmax(scores; dims=2)
    out = dropdims(sum(reshape(weights, 1, H, np, n) .* v; dims=3); dims=3) # (hd, H, n)
    pooled = h.wo(reshape(out, hidden, n))            # (hidden, n)

    return pooled .+ h.mlp(h.layernorm(pooled))
end

Flux.@layer SiglipAttentionPoolingHead

"""
    SiglipVisionModel{E, L, N, H}

embeddings → pre-norm encoder → `post_layernorm` → attention-pooling head.
Returns the image embedding `(hidden, batch)`.
"""
struct SiglipVisionModel{E,L,N,H}
    embeddings::E
    layers::L
    post_layernorm::N
    head::H
end

function (m::SiglipVisionModel)(pixel_values::AbstractArray{<:Real,4})
    h = m.embeddings(pixel_values)
    for layer in m.layers
        h = layer(h)
    end
    return m.head(m.post_layernorm(h))
end

Flux.@layer SiglipVisionModel

"""
    SiglipTextModel{E, P, L, N, H}

token + position embeddings → pre-norm encoder → `final_layer_norm` → last-token
pooling → linear `head`. Returns the text embedding `(projection_size, batch)`.
"""
struct SiglipTextModel{E,P,L,N,H}
    token_embedding::E
    position_embeddings::P
    layers::L
    final_layer_norm::N
    head::H
end

function (m::SiglipTextModel)(input_ids::AbstractMatrix{<:Integer})
    seq = size(input_ids, 1)
    h = m.token_embedding(input_ids)                  # (hidden, seq, n)
    pos = reshape(m.position_embeddings[:, 1:seq], size(h, 1), seq, 1)
    h = h .+ pos
    for layer in m.layers
        h = layer(h)
    end
    h = m.final_layer_norm(h)
    return m.head(h[:, end, :])                        # last token → (proj, n)
end

Flux.@layer SiglipTextModel

"""
    SiglipModel{C, V, T}

Vision + text towers with a sigmoid-contrastive head. `logit_scale` and
`logit_bias` are 1-element buffers; forward returns `logits_per_image`
`(n_images, n_texts)`.
"""
struct SiglipModel{C,V,T}
    config::C
    vision::V
    text::T
    logit_scale::Vector{Float32}
    logit_bias::Vector{Float32}
end

function (m::SiglipModel)(
    pixel_values::AbstractArray{<:Real,4}, input_ids::AbstractMatrix{<:Integer}
)
    img = l2_normalize(m.vision(pixel_values))         # (proj, n_img)
    txt = l2_normalize(m.text(input_ids))              # (proj, n_txt)
    scale = exp(m.logit_scale[1])
    return (permutedims(img) * txt) .* scale .+ m.logit_bias[1]   # (n_img, n_txt)
end

Flux.@layer SiglipModel

function SiglipModel(cfg::SiglipConfig)
    v, t = cfg.vision, cfg.text
    veps = Float32(v.layer_norm_eps)
    teps = Float32(t.layer_norm_eps)
    v_hd = v.hidden_size ÷ v.num_attention_heads
    num_patches = (v.image_size ÷ v.patch_size)^2
    patch_dim = v.num_channels * v.patch_size^2

    vision = SiglipVisionModel(
        SiglipVisionEmbeddings(
            ViTPatchEmbeddings(
                Linear(patch_dim, v.hidden_size; bias=true), v.patch_size, v.num_channels
            ),
            zeros(Float32, v.hidden_size, num_patches),
        ),
        [
            _siglip_layer(v.hidden_size, v.num_attention_heads, v.intermediate_size, veps)
            for _ in 1:(v.num_hidden_layers)
        ],
        LayerNorm(v.hidden_size, veps),
        SiglipAttentionPoolingHead(
            Linear(v.hidden_size, v.hidden_size; bias=true),
            Linear(v.hidden_size, v.hidden_size; bias=true),
            Linear(v.hidden_size, v.hidden_size; bias=true),
            Linear(v.hidden_size, v.hidden_size; bias=true),
            zeros(Float32, v.hidden_size),
            LayerNorm(v.hidden_size, veps),
            GeluMLP(v.hidden_size, v.intermediate_size; approx=true),
            v.num_attention_heads,
            v_hd,
        ),
    )

    text = SiglipTextModel(
        TokenEmbedding(t.vocab_size, t.hidden_size),
        zeros(Float32, t.hidden_size, t.max_position_embeddings),
        [
            _siglip_layer(t.hidden_size, t.num_attention_heads, t.intermediate_size, teps)
            for _ in 1:(t.num_hidden_layers)
        ],
        LayerNorm(t.hidden_size, teps),
        Linear(t.hidden_size, t.projection_size; bias=true),
    )

    return SiglipModel(cfg, vision, text, zeros(Float32, 1), zeros(Float32, 1))
end

# Fill the shared pre-norm encoder-layer entries for a tower under `root`.
function _siglip_encoder_map!(out, hf::AbstractString, root::Tuple, n_layers::Integer)
    for i in 0:(n_layers - 1)
        lp = (root..., :layers, i + 1)
        p = "$(hf).encoder.layers.$(i)"
        for (proj, dst) in
            (("q_proj", :wq), ("k_proj", :wk), ("v_proj", :wv), ("out_proj", :wo))
            out["$(p).self_attn.$(proj).weight"] = (
                (lp..., :self_attn, dst, :weight), :as_is
            )
            out["$(p).self_attn.$(proj).bias"] = ((lp..., :self_attn, dst, :bias), :as_is)
        end
        out["$(p).layer_norm1.weight"] = ((lp..., :input_layernorm, :weight), :as_is)
        out["$(p).layer_norm1.bias"] = ((lp..., :input_layernorm, :bias), :as_is)
        out["$(p).layer_norm2.weight"] = (
            (lp..., :post_attention_layernorm, :weight), :as_is
        )
        out["$(p).layer_norm2.bias"] = ((lp..., :post_attention_layernorm, :bias), :as_is)
        out["$(p).mlp.fc1.weight"] = ((lp..., :mlp, :c_fc, :weight), :as_is)
        out["$(p).mlp.fc1.bias"] = ((lp..., :mlp, :c_fc, :bias), :as_is)
        out["$(p).mlp.fc2.weight"] = ((lp..., :mlp, :c_proj, :weight), :as_is)
        out["$(p).mlp.fc2.bias"] = ((lp..., :mlp, :c_proj, :bias), :as_is)
    end
    return out
end

"""
    siglip_state_dict_map(cfg::SiglipConfig) -> Dict{String, Tuple{Tuple, Symbol}}

Everything except the tensors that need reshaping (patch conv, the pooling
head's fused `in_proj` and `probe`, and the scalar logits), which
`load_state_dict!` handles by hand. Embedding tables load with `:transpose`.
"""
function siglip_state_dict_map(cfg::SiglipConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    out["vision_model.embeddings.patch_embedding.bias"] = (
        (:vision, :embeddings, :patch_embeddings, :projection, :bias), :as_is
    )
    out["vision_model.embeddings.position_embedding.weight"] = (
        (:vision, :embeddings, :position_embeddings), :transpose
    )
    _siglip_encoder_map!(out, "vision_model", (:vision,), cfg.vision.num_hidden_layers)
    out["vision_model.post_layernorm.weight"] = (
        (:vision, :post_layernorm, :weight), :as_is
    )
    out["vision_model.post_layernorm.bias"] = ((:vision, :post_layernorm, :bias), :as_is)
    out["vision_model.head.attention.out_proj.weight"] = (
        (:vision, :head, :wo, :weight), :as_is
    )
    out["vision_model.head.attention.out_proj.bias"] = (
        (:vision, :head, :wo, :bias), :as_is
    )
    out["vision_model.head.layernorm.weight"] = (
        (:vision, :head, :layernorm, :weight), :as_is
    )
    out["vision_model.head.layernorm.bias"] = ((:vision, :head, :layernorm, :bias), :as_is)
    out["vision_model.head.mlp.fc1.weight"] = (
        (:vision, :head, :mlp, :c_fc, :weight), :as_is
    )
    out["vision_model.head.mlp.fc1.bias"] = ((:vision, :head, :mlp, :c_fc, :bias), :as_is)
    out["vision_model.head.mlp.fc2.weight"] = (
        (:vision, :head, :mlp, :c_proj, :weight), :as_is
    )
    out["vision_model.head.mlp.fc2.bias"] = ((:vision, :head, :mlp, :c_proj, :bias), :as_is)

    out["text_model.embeddings.token_embedding.weight"] = (
        (:text, :token_embedding, :weight), :transpose
    )
    out["text_model.embeddings.position_embedding.weight"] = (
        (:text, :position_embeddings), :transpose
    )
    _siglip_encoder_map!(out, "text_model", (:text,), cfg.text.num_hidden_layers)
    out["text_model.final_layer_norm.weight"] = (
        (:text, :final_layer_norm, :weight), :as_is
    )
    out["text_model.final_layer_norm.bias"] = ((:text, :final_layer_norm, :bias), :as_is)
    out["text_model.head.weight"] = ((:text, :head, :weight), :as_is)
    out["text_model.head.bias"] = ((:text, :head, :bias), :as_is)

    return out
end

"""
    load_state_dict!(m::SiglipModel, weights) -> m

Load a SigLIP checkpoint. Most parameters go through
[`siglip_state_dict_map`](@ref); the patch conv `(hidden, C, P, P)` is reshaped
to its `Linear`, the pooling head's fused `in_proj` `[Q; K; V]` is split into
`wq`/`wk`/`wv`, the `(1,1,H)` probe is squeezed, and the scalar
`logit_scale`/`logit_bias` are copied in.
"""
function load_state_dict!(m::SiglipModel, weights::AbstractDict{String,<:AbstractArray})
    load_into!(m, weights, siglip_state_dict_map(m.config))

    vhidden = m.config.vision.hidden_size
    conv = weights["vision_model.embeddings.patch_embedding.weight"]   # (hidden, C, P, P)
    copyto!(
        m.vision.embeddings.patch_embeddings.projection.weight,
        reshape(Array(conv), vhidden, :),
    )

    head = m.vision.head
    copyto!(head.probe, vec(weights["vision_model.head.probe"]))
    Wi = weights["vision_model.head.attention.in_proj_weight"]         # (3h, h), [Q; K; V]
    bi = weights["vision_model.head.attention.in_proj_bias"]           # (3h,)
    copyto!(head.wq.weight, Wi[1:vhidden, :])
    copyto!(head.wk.weight, Wi[(vhidden + 1):(2vhidden), :])
    copyto!(head.wv.weight, Wi[(2vhidden + 1):(3vhidden), :])
    copyto!(head.wq.bias, bi[1:vhidden])
    copyto!(head.wk.bias, bi[(vhidden + 1):(2vhidden)])
    copyto!(head.wv.bias, bi[(2vhidden + 1):(3vhidden)])

    m.logit_scale[1] = only(weights["logit_scale"])
    m.logit_bias[1] = only(weights["logit_bias"])
    return m
end
