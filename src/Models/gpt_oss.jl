"""
    GptOssConfig

Architectural hyperparameters for OpenAI's gpt-oss (`gpt-oss-20b`, `gpt-oss-120b`).
Field names mirror HuggingFace's `config.json` keys.

What sets the family apart, all handled here:

- Attention carries a learned sink per query head, biases on all four
  projections, and alternates sliding-window (`sliding_window` tokens) and full
  layers as `layer_types` lists them.
- RoPE uses YaRN (`rope_scaling`, a [`YarnRopeScaling`](@ref)) to stretch a 4k
  training context to 128k.
- The MLP is a mixture of `num_local_experts` experts, `num_experts_per_tok` of
  them per token, chosen by a biased router and weighted by a softmax over the
  chosen logits. Each expert is a SwiGLU whose gate and up projections are
  interleaved and clamped to `swiglu_limit`.
"""
Base.@kwdef struct GptOssConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    num_local_experts::Int
    num_experts_per_tok::Int
    max_position_embeddings::Int
    rope_theta::Float64 = 150_000.0
    rope_scaling::Union{Nothing,YarnRopeScaling} = nothing
    rms_norm_eps::Float64 = 1.0e-5
    sliding_window::Int = 128
    # HF's default alternates, starting with a sliding layer.
    layer_types::Vector{Symbol} = [
        isodd(i) ? :sliding_attention : :full_attention for i in 1:num_hidden_layers
    ]
    attention_bias::Bool = true
    swiglu_limit::Float64 = 7.0
    tie_word_embeddings::Bool = false
end

const _FP4_E2M1 = Float32[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

"""
    MXFP4Weights(blocks, scales)

A stack of expert weight matrices in OCP MXFP4, held exactly as a checkpoint
stores them. Each output row is packed along the input dimension in blocks of 32
four-bit E2M1 values (two per byte, low nibble first) sharing one power-of-two
scale, stored as an exponent biased by 127.

`blocks` is `(16, n_blocks, n_out, n_experts)` and `scales` is
`(n_blocks, n_out, n_experts)`: the file's byte order, which a lazy
[`load_weights`](@ref) view exposes without copying. An expert is decoded into
`Float32` only when it runs, so gpt-oss-20b's experts take about 13 GB instead of
the 50 GB they would as `Float32`.
"""
struct MXFP4Weights{B<:AbstractArray{UInt8,4},S<:AbstractArray{UInt8,3}}
    blocks::B
    scales::S
    function MXFP4Weights(blocks::AbstractArray{UInt8,4}, scales::AbstractArray{UInt8,3})
        size(blocks, 1) == 16 ||
            throw(DimensionMismatch("MXFP4 blocks hold 16 bytes, got $(size(blocks, 1))"))
        size(blocks)[2:4] == size(scales) ||
            throw(DimensionMismatch("blocks $(size(blocks)) and scales $(size(scales)) disagree"))
        return new{typeof(blocks),typeof(scales)}(blocks, scales)
    end
end

_n_out(w::MXFP4Weights) = size(w.blocks, 3)
_n_in(w::MXFP4Weights) = 32 * size(w.blocks, 2)
_n_experts(w::MXFP4Weights) = size(w.blocks, 4)

"""
    decode_mxfp4!(buffer, w::MXFP4Weights, expert) -> buffer

Decode one expert into `buffer`, transposed: `buffer[k, r]` is input `k`'s
weight for output `r`. Each output row decodes into one contiguous column, and
`transpose(buffer) * x` applies the expert.
"""
function decode_mxfp4!(buffer::AbstractMatrix{Float32}, w::MXFP4Weights, expert::Integer)
    size(buffer) == (_n_in(w), _n_out(w)) ||
        throw(DimensionMismatch("buffer $(size(buffer)), expected $((_n_in(w), _n_out(w)))"))
    blocks, scales, e = w.blocks, w.scales, Int(expert)
    Threads.@threads for r in 1:_n_out(w)
        @inbounds for g in axes(blocks, 2)
            exponent = Int(scales[g, r, e]) - 127
            offset = 32 * (g - 1)
            for b in 1:16
                byte = blocks[b, g, r, e]
                buffer[offset + 2b - 1, r] = ldexp(_FP4_E2M1[(byte & 0x0f) + 1], exponent)
                buffer[offset + 2b, r] = ldexp(_FP4_E2M1[(byte >> 4) + 1], exponent)
            end
        end
    end
    return buffer
end

"""
    GptOssExperts

gpt-oss's expert stack. `gate_up` maps the hidden state to interleaved gate and
up activations (`2 * intermediate` outputs, gate at odd rows) and `down` maps back.
Each is either a dense `(n_out, n_in, n_experts)` array or [`MXFP4Weights`](@ref);
the biases are `(n_out, n_experts)`.
"""
struct GptOssExperts{W1,B1,W2,B2}
    gate_up::W1
    gate_up_bias::B1
    down::W2
    down_bias::B2
    alpha::Float32
    limit::Float32
end

Flux.@layer GptOssExperts

function GptOssExperts(hidden::Integer, intermediate::Integer, n_experts::Integer; limit::Real=7.0)
    init = Flux.glorot_uniform
    return GptOssExperts(
        init(2intermediate, hidden, n_experts), zeros(Float32, 2intermediate, n_experts),
        init(hidden, intermediate, n_experts), zeros(Float32, hidden, n_experts),
        1.702f0, Float32(limit),
    )
end

_expert_buffer(::AbstractArray{<:Real,3}) = nothing
_expert_buffer(w::MXFP4Weights) = Matrix{Float32}(undef, _n_in(w), _n_out(w))

_apply_expert(w::AbstractArray{<:Real,3}, e::Int, x, _) = view(w, :, :, e) * x
_apply_expert(w::MXFP4Weights, e::Int, x, buffer) = transpose(decode_mxfp4!(buffer, w, e)) * x

# HF's `_apply_gate`: gate and up interleave, the gate is capped above and the up
# projection clamped both ways, then SiLU with a sharpened sigmoid times (up + 1).
function _clamped_swiglu(gate_up::AbstractMatrix, alpha::Float32, limit::Float32)
    gate = min.(view(gate_up, 1:2:size(gate_up, 1), :), limit)
    up = clamp.(view(gate_up, 2:2:size(gate_up, 1), :), -limit, limit)
    return (up .+ 1) .* (gate .* NNlib.sigmoid.(gate .* alpha))
end

"""
    GptOssMLP

Router plus [`GptOssExperts`](@ref). Each token goes to its `top_k` experts by
router logit, and their outputs are weighted by a softmax over those `top_k`
logits. Experts run in ascending order and accumulate into the output, the
order HF uses, so the summation matches too. Inference only: routing and the
decoded-expert buffers are not differentiable.
"""
struct GptOssMLP{R,E}
    router::R
    experts::E
    top_k::Int
end

Flux.@layer GptOssMLP

function (m::GptOssMLP)(x::AbstractArray)
    hidden = size(x, 1)
    trailing = size(x)[2:end]
    n_tokens = prod(trailing)
    flat = reshape(x, hidden, n_tokens)

    logits = Array(m.router(flat))                    # (n_experts, n_tokens)
    n_experts = size(logits, 1)
    assigned = [Int[] for _ in 1:n_experts]
    weights = [Float32[] for _ in 1:n_experts]
    for t in 1:n_tokens
        chosen = partialsortperm(view(logits, :, t), 1:(m.top_k); rev=true)
        top = logits[chosen, t]
        probs = exp.(top .- maximum(top))
        probs ./= sum(probs)
        for (e, p) in zip(chosen, probs)
            push!(assigned[e], t)
            push!(weights[e], p)
        end
    end

    ex = m.experts
    gate_up_buffer = _expert_buffer(ex.gate_up)
    down_buffer = _expert_buffer(ex.down)
    out = zeros(Float32, hidden, n_tokens)
    for e in 1:n_experts
        tokens = assigned[e]
        isempty(tokens) && continue
        xs = flat[:, tokens]
        gate_up = _apply_expert(ex.gate_up, e, xs, gate_up_buffer) .+ view(ex.gate_up_bias, :, e)
        act = _clamped_swiglu(gate_up, ex.alpha, ex.limit)
        y = _apply_expert(ex.down, e, act, down_buffer) .+ view(ex.down_bias, :, e)
        out[:, tokens] .+= y .* reshape(weights[e], 1, :)
    end
    return reshape(out, hidden, trailing...)
end

"""
    GptOssForCausalLM

gpt-oss: a [`DecoderModel`](@ref) trunk whose layers use sink attention and
[`GptOssMLP`](@ref), plus an untied LM head.

`GptOssForCausalLM(cfg)` builds a randomly initialized model with dense experts,
which suits small test configs. To load a checkpoint, use
[`load_pretrained`](@ref) (or [`HuggingFaceTransformers.load`](@ref)), which
calls `GptOssForCausalLM(cfg, weights)`: MXFP4 experts then stay as the file's
bytes, memory-mapped, and only the remaining weights are allocated.
"""
struct GptOssForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

Flux.@layer GptOssForCausalLM
function Flux.Optimisers.trainable(lm::GptOssForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

function (lm::GptOssForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    return lm.lm_head(lm.model(input_ids; kwargs...))
end

function _gpt_oss_attention(cfg::GptOssConfig, rope, layer::Int)
    kind = cfg.layer_types[layer]
    kind in (:sliding_attention, :full_attention) ||
        throw(ArgumentError("unsupported gpt-oss layer type `$(kind)`"))
    return GQA(
        cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, rope;
        window_size=kind === :sliding_attention ? cfg.sliding_window : nothing,
        qkv_bias=cfg.attention_bias, wo_bias=cfg.attention_bias, sinks=true,
    )
end

function _gpt_oss_model(cfg::GptOssConfig, experts::AbstractVector)
    length(cfg.layer_types) == cfg.num_hidden_layers || throw(
        ArgumentError(
            "layer_types lists $(length(cfg.layer_types)) layers for $(cfg.num_hidden_layers)",
        ),
    )
    rope = RoPE(cfg.head_dim; base=cfg.rope_theta, yarn=cfg.rope_scaling)
    eps = Float32(cfg.rms_norm_eps)
    layers = [
        DecoderLayer(
            _gpt_oss_attention(cfg, rope, i),
            RMSNorm(cfg.hidden_size, eps),
            RMSNorm(cfg.hidden_size, eps),
            GptOssMLP(
                Linear(cfg.hidden_size, cfg.num_local_experts; bias=true),
                experts[i],
                cfg.num_experts_per_tok,
            ),
        ) for i in 1:(cfg.num_hidden_layers)
    ]
    model = DecoderModel(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size), layers, RMSNorm(cfg.hidden_size, eps)
    )
    return GptOssForCausalLM(cfg, model, Linear(cfg.hidden_size, cfg.vocab_size))
end

function GptOssForCausalLM(cfg::GptOssConfig)
    experts = [
        GptOssExperts(
            cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts;
            limit=cfg.swiglu_limit,
        ) for _ in 1:(cfg.num_hidden_layers)
    ]
    return _gpt_oss_model(cfg, experts)
end

_expert_key(i::Int, name::String) = "model.layers.$(i - 1).mlp.experts.$(name)"

# One layer's expert projection straight from a checkpoint: MXFP4 blocks and
# scales as the file holds them, or a dense `(n_experts, n_in, n_out)` tensor,
# which HF applies as `x @ W` and Julia as `W * x`, hence the reversed dims.
function _checkpoint_experts(weights, i::Int, proj::String)
    blocks = _expert_key(i, proj * "_blocks")
    if haskey(weights, blocks)
        return MXFP4Weights(
            _file_order(weights[blocks]), _file_order(weights[_expert_key(i, proj * "_scales")])
        )
    end
    return Array{Float32}(_file_order(weights[_expert_key(i, proj)]))
end

"""
    GptOssForCausalLM(cfg, weights)

Build gpt-oss directly from an HF-keyed weight dictionary, as returned by
[`load_weights`](@ref). Expert weights are taken as they are, MXFP4 or dense,
rather than copied into preallocated arrays; everything else loads through
[`gpt_oss_state_dict_map`](@ref).
"""
function GptOssForCausalLM(cfg::GptOssConfig, weights::AbstractDict{String,<:AbstractArray})
    experts = map(1:(cfg.num_hidden_layers)) do i
        GptOssExperts(
            _checkpoint_experts(weights, i, "gate_up_proj"),
            zeros(Float32, 2cfg.intermediate_size, cfg.num_local_experts),
            _checkpoint_experts(weights, i, "down_proj"),
            zeros(Float32, cfg.hidden_size, cfg.num_local_experts),
            1.702f0,
            Float32(cfg.swiglu_limit),
        )
    end
    # Layers mixing MXFP4 and dense experts would not share a type.
    lm = _gpt_oss_model(cfg, identity.(experts))
    load_into!(lm, weights, gpt_oss_state_dict_map(cfg))
    return _tie_lm_head!(lm)
end

function _tie_lm_head!(lm::GptOssForCausalLM)
    if lm.config.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end
    return lm
end

"""
    gpt_oss_state_dict_map(cfg::GptOssConfig) -> Dict

HuggingFace → internal parameter table for gpt-oss, without the expert weight
tensors themselves, which [`GptOssForCausalLM`](@ref) takes from the checkpoint
as they are. Expert biases are included: HF stores them `(n_experts, n_out)`.
"""
function gpt_oss_state_dict_map(cfg::GptOssConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()
    out["model.embed_tokens.weight"] = ((:model, :embed_tokens, :weight), :transpose)
    out["model.norm.weight"] = ((:model, :norm, :weight), :as_is)
    cfg.tie_word_embeddings || (out["lm_head.weight"] = ((:lm_head, :weight), :as_is))
    for i in 0:(cfg.num_hidden_layers - 1)
        layer = (:model, :layers, i + 1)
        prefix = "model.layers.$(i)"
        out["$(prefix).input_layernorm.weight"] = ((layer..., :input_layernorm, :weight), :as_is)
        out["$(prefix).post_attention_layernorm.weight"] =
            ((layer..., :post_attention_layernorm, :weight), :as_is)
        for (hf, field) in (("q_proj", :wq), ("k_proj", :wk), ("v_proj", :wv), ("o_proj", :wo))
            out["$(prefix).self_attn.$(hf).weight"] = ((layer..., :self_attn, field, :weight), :as_is)
            if cfg.attention_bias
                out["$(prefix).self_attn.$(hf).bias"] = ((layer..., :self_attn, field, :bias), :as_is)
            end
        end
        out["$(prefix).self_attn.sinks"] = ((layer..., :self_attn, :sinks), :as_is)
        out["$(prefix).mlp.router.weight"] = ((layer..., :mlp, :router, :weight), :as_is)
        out["$(prefix).mlp.router.bias"] = ((layer..., :mlp, :router, :bias), :as_is)
        out["$(prefix).mlp.experts.gate_up_proj_bias"] =
            ((layer..., :mlp, :experts, :gate_up_bias), :transpose)
        out["$(prefix).mlp.experts.down_proj_bias"] =
            ((layer..., :mlp, :experts, :down_bias), :transpose)
    end
    return out
end

"""
    load_state_dict!(lm::GptOssForCausalLM, weights) -> lm

Load a checkpoint into an existing model. Dense expert weights are copied in;
MXFP4 experts are decoded into the model's dense arrays, which is only sensible
for small models; [`GptOssForCausalLM(cfg, weights)`](@ref GptOssForCausalLM)
keeps them packed instead.
"""
function load_state_dict!(lm::GptOssForCausalLM, weights::AbstractDict{String,<:AbstractArray})
    load_into!(lm, weights, gpt_oss_state_dict_map(lm.config))
    for (i, layer) in enumerate(lm.model.layers)
        ex = layer.mlp.experts
        for (dst, proj) in ((ex.gate_up, "gate_up_proj"), (ex.down, "down_proj"))
            src = _checkpoint_experts(weights, i, proj)
            if src isa MXFP4Weights
                buffer = _expert_buffer(src)
                for e in 1:_n_experts(src)
                    dst[:, :, e] .= transpose(decode_mxfp4!(buffer, src, e))
                end
            else
                copyto!(dst, src)
            end
        end
    end
    return _tie_lm_head!(lm)
end

"""
    build_caches(lm::GptOssForCausalLM, max_seq, batch_size; eltype=Float32)

Allocate one [`KVCache`](@ref) per decoder layer. Sliding layers keep every
position too; their window is applied by the attention mask.
"""
function build_caches(
    lm::GptOssForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    proto = lm.model.embed_tokens.weight
    return [
        KVCache(proto, cfg.head_dim, cfg.num_key_value_heads, max_seq, batch_size; eltype=eltype)
        for _ in 1:(cfg.num_hidden_layers)
    ]
end
