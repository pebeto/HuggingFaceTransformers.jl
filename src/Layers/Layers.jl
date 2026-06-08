"""
    Allspark.Layers

The minimal layer kit used to assemble transformer models: `RMSNorm`,
`RoPE` (with Llama-3 scaling hooks), bias-less `Linear`, SiLU-gated MLP,
grouped-query attention with KV-cache support. All layers are
`Flux.@layer`-annotated and `Functors.fmap`-friendly.
"""
module Layers

using Flux
using NNlib
using LinearAlgebra
using Statistics

export KVCache,
    RMSNorm,
    GemmaRMSNorm,
    LayerNorm,
    RoPE,
    Linear,
    SiLUGatedMLP,
    GeluGatedMLP,
    GeluMLP,
    MoEMLP,
    GQA,
    softcap,
    reset!

"""
    KVCache{T <: AbstractArray}

A mutable KV-cache wrapper holding preallocated keys and values for
autoregressive generation. Both fields are 4-D arrays of shape
`(head_dim, n_kv_heads, max_seq, batch_size)` — feature-first to match the
projection layout used in `GQA`, so cache writes/reads are straight slice
assignments with no `permutedims`.
"""
mutable struct KVCache{T<:AbstractArray}
    k::T
    v::T
end

"""
    KVCache(head_dim::Integer, n_kv_heads::Integer, max_seq::Integer, batch_size::Integer; eltype = Float32)

Create a preallocated `KVCache` of shape
`(head_dim, n_kv_heads, max_seq, batch_size)` with element type `eltype`.
"""
function KVCache(
    head_dim::Integer,
    n_kv_heads::Integer,
    max_seq::Integer,
    batch_size::Integer;
    eltype=Float32,
)
    k = zeros(eltype, head_dim, n_kv_heads, max_seq, batch_size)
    v = zeros(eltype, head_dim, n_kv_heads, max_seq, batch_size)
    return KVCache(k, v)
end

Flux.@layer KVCache
Flux.Optimisers.trainable(::KVCache) = (;)

Base.eltype(::KVCache{T}) where {T} = eltype(T)
Base.size(c::KVCache) = size(c.k)

function Base.show(io::IO, c::KVCache)
    h, k, s, b = size(c.k)
    return print(io, "KVCache(", h, "×", k, "×", s, "×", b, ", ", eltype(c), ")")
end

"""
    reset!(cache::KVCache) -> cache

Zero the cache's `k` and `v` arrays in place so the same allocation can
be reused across generations. Returns the cache for chaining.
"""
function reset!(cache::KVCache)
    fill!(cache.k, 0)
    fill!(cache.v, 0)
    return cache
end

"""
    reset!(caches::AbstractVector{<:KVCache}) -> caches

Reset a per-layer cache bundle (the vector returned by `build_caches`).
"""
function reset!(caches::AbstractVector{<:KVCache})
    for c in caches
        reset!(c)
    end
    return caches
end

"""
    RMSNorm{W, T}

Root Mean Square Normalization layer. Normalizes inputs along the first dimension (features/channels).
\$\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\text{mean}(x^2) + \\epsilon}} \\cdot w\$
"""
struct RMSNorm{W,T}
    weight::W
    eps::T
end

"""
    RMSNorm(dim::Integer, eps::Real = 1f-6)

Construct an `RMSNorm` layer of dimension `dim` with stability parameter `eps`.
"""
function RMSNorm(dim::Integer, eps::Real=1.0f-6)
    weight = ones(Float32, dim)
    return RMSNorm(weight, Float32(eps))
end

function (m::RMSNorm)(x::AbstractArray)
    rms = sqrt.(sum(x .^ 2; dims=1) ./ size(x, 1) .+ m.eps)
    return (x ./ rms) .* m.weight
end

Flux.@layer RMSNorm
Flux.Optimisers.trainable(m::RMSNorm) = (; weight=m.weight)

"""
    GemmaRMSNorm{W, T}

Gemma2's RMSNorm variant: scales by `(1 + weight)` instead of just
`weight`. The off-by-one matters for numerical parity — HF's checkpoint
weights are stored under the assumption of this scaling, so substituting
plain `RMSNorm` yields drifted logits.
"""
struct GemmaRMSNorm{W,T}
    weight::W
    eps::T
end

"""
    GemmaRMSNorm(dim::Integer, eps::Real = 1f-6)

Construct a `GemmaRMSNorm` layer of dimension `dim`. The weight starts
at zero (so `(1 + weight) = 1` and the norm is identity-scaled before
loading from the checkpoint).
"""
function GemmaRMSNorm(dim::Integer, eps::Real=1.0f-6)
    weight = zeros(Float32, dim)
    return GemmaRMSNorm(weight, Float32(eps))
end

function (m::GemmaRMSNorm)(x::AbstractArray)
    rms = sqrt.(sum(x .^ 2; dims=1) ./ size(x, 1) .+ m.eps)
    return (x ./ rms) .* (1 .+ m.weight)
end

Flux.@layer GemmaRMSNorm
Flux.Optimisers.trainable(m::GemmaRMSNorm) = (; weight=m.weight)

"""
    LayerNorm{W, B, T}

Standard pre-activation Layer Normalization: subtract the per-token mean,
divide by the per-token standard deviation, then scale by `weight` and
shift by `bias`. Used by GPT-2, GPT-NeoX, BERT-family, and anything else
that predates the RMSNorm convention.
"""
struct LayerNorm{W,B,T}
    weight::W
    bias::B
    eps::T
end

"""
    LayerNorm(dim::Integer, eps::Real = 1f-5)

Construct a `LayerNorm` layer of dimension `dim`. Weight initializes to
ones, bias to zeros — the identity at load time, matching how HF stores
checkpoints.
"""
function LayerNorm(dim::Integer, eps::Real=1.0f-5)
    weight = ones(Float32, dim)
    bias = zeros(Float32, dim)
    return LayerNorm(weight, bias, Float32(eps))
end

function (m::LayerNorm)(x::AbstractArray)
    μ = mean(x; dims=1)
    σ² = sum((x .- μ) .^ 2; dims=1) ./ size(x, 1)
    return ((x .- μ) ./ sqrt.(σ² .+ m.eps)) .* m.weight .+ m.bias
end

Flux.@layer LayerNorm
Flux.Optimisers.trainable(m::LayerNorm) = (; weight=m.weight, bias=m.bias)

"""
    softcap(x, cap)

`cap * tanh(x / cap)`. Used by Gemma2 to bound attention scores
(`attn_logit_softcapping`) and final logits (`final_logit_softcapping`).
Returns `x` unchanged when `cap` is `nothing`.
"""
softcap(x, ::Nothing) = x
softcap(x, cap::Real) = oftype(x, cap) * tanh(x / oftype(x, cap))
softcap(x::AbstractArray, cap::Real) = Float32(cap) .* tanh.(x ./ Float32(cap))
softcap(x::AbstractArray, ::Nothing) = x

"""
    RoPE{F}

Rotary Position Embeddings (RoPE) layer. Computes positional rotations using frequency buffers `inv_freq`.
Supports Llama 3 scaling/wavelength-interpolation if constructed with the appropriate scaling parameters.
"""
struct RoPE{F}
    inv_freq::F
    rotary_dim::Union{Nothing,Int}
end

"""
    RoPE(dim::Integer; base::Real = 500000.0, rotary_dim = nothing, scaling_factor = nothing, low_freq_factor = nothing, high_freq_factor = nothing, old_context_len = nothing)

Construct a `RoPE` layer. `dim` is the full attention `head_dim`.

`rotary_dim` controls partial rotary embeddings: when set (e.g. 16 for
Pythia/GPT-NeoX with `rotary_pct=0.25` over a 64-dim head), only the
first `rotary_dim` channels of each head are rotated and the rest pass
through. When `nothing` (the default) the full head is rotated.

If scaling parameters are provided, applies Llama 3 NTK-style frequency
scaling.
"""
function RoPE(
    dim::Integer;
    base::Real=500000.0,
    rotary_dim::Union{Nothing,Integer}=nothing,
    scaling_factor::Union{Nothing,Real}=nothing,
    low_freq_factor::Union{Nothing,Real}=nothing,
    high_freq_factor::Union{Nothing,Real}=nothing,
    old_context_len::Union{Nothing,Integer}=nothing,
)
    # Compute inv_freq against the rotated portion only — partial RoPE
    # keeps the unrotated tail unchanged, so its frequencies don't matter.
    rdim = isnothing(rotary_dim) ? Int(dim) : Int(rotary_dim)
    indices = 0:2:(rdim - 2)
    inv_freq = 1.0 ./ (base .^ (indices ./ rdim))

    # Apply Llama 3/3.1 RoPE frequency scaling if all scaling parameters are provided
    if !isnothing(scaling_factor) &&
        !isnothing(low_freq_factor) &&
        !isnothing(high_freq_factor) &&
        !isnothing(old_context_len)
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * pi ./ inv_freq
        inv_freq_scaled = copy(inv_freq)
        for i in eachindex(inv_freq)
            wl = wavelen[i]
            if wl > low_freq_wavelen
                inv_freq_scaled[i] = inv_freq[i] / scaling_factor
            elseif wl >= high_freq_wavelen
                smooth =
                    (old_context_len / wl - low_freq_factor) /
                    (high_freq_factor - low_freq_factor)
                inv_freq_scaled[i] =
                    (1 - smooth) * (inv_freq[i] / scaling_factor) + smooth * inv_freq[i]
            end
        end
        inv_freq = inv_freq_scaled
    end

    stored_rdim = isnothing(rotary_dim) ? nothing : Int(rotary_dim)
    return RoPE(Float32.(inv_freq), stored_rdim)
end

# Core rotation: assumes inv_freq matches `size(x, 1)`. Pulled out so
# partial-rotary mode can call it on a slice of the input.
function _rope_rotate(
    inv_freq::AbstractVector, x::AbstractArray, position_ids::AbstractVector
)
    # Compute rotation angles: thetas of shape (half_dim, seq_len)
    thetas = inv_freq * reshape(position_ids, 1, :)
    cos_half = cos.(thetas)
    sin_half = sin.(thetas)

    # Duplicate cos/sin for the two halves of head_dim
    cos_emb_2d = vcat(cos_half, cos_half)
    sin_emb_2d = vcat(sin_half, sin_half)

    # Reshape cos/sin to match shape of x
    if ndims(x) == 4
        cos_emb = reshape(cos_emb_2d, (size(x, 1), 1, size(x, 3), 1))
        sin_emb = reshape(sin_emb_2d, (size(x, 1), 1, size(x, 3), 1))
    elseif ndims(x) == 3
        cos_emb = reshape(cos_emb_2d, (size(x, 1), size(x, 2), 1))
        sin_emb = reshape(sin_emb_2d, (size(x, 1), size(x, 2), 1))
    else
        cos_emb = cos_emb_2d
        sin_emb = sin_emb_2d
    end

    # Apply rotation
    half_dim = size(x, 1) ÷ 2
    x1 = selectdim(x, 1, 1:half_dim)
    x2 = selectdim(x, 1, (half_dim + 1):size(x, 1))
    x_rotated = vcat(-x2, x1)

    return (x .* cos_emb) .+ (x_rotated .* sin_emb)
end

function (m::RoPE)(x::AbstractArray, position_ids::AbstractVector)
    # Full rotation path: either rotary_dim is unset, or it equals head_dim.
    if isnothing(m.rotary_dim) || m.rotary_dim == size(x, 1)
        return _rope_rotate(m.inv_freq, x, position_ids)
    end

    # Partial RoPE (GPT-NeoX / Pythia): rotate the first rotary_dim
    # channels of each head, pass the remaining channels through.
    rdim = m.rotary_dim
    x_rot = Array(selectdim(x, 1, 1:rdim))
    x_pass = Array(selectdim(x, 1, (rdim + 1):size(x, 1)))
    rotated = _rope_rotate(m.inv_freq, x_rot, position_ids)
    return cat(rotated, x_pass; dims=1)
end

Flux.@layer RoPE
Flux.Optimisers.trainable(::RoPE) = (;)

"""
    Linear{W, B}

A linear projection layer with optional bias. `B` is `Nothing` for a
bias-less layer (Llama / Mistral convention) or an `AbstractVector` when
a bias is present (Qwen Q/K/V projections, BERT linears, …).

Forward and `trainable` dispatch on `B` so the no-bias path stays free
of runtime branching.
"""
struct Linear{W,B}
    weight::W
    bias::B
end

"""
    Linear(in_features::Integer, out_features::Integer; bias=false, init = Flux.glorot_uniform)

Construct a `Linear` layer with input dimension `in_features` and output
dimension `out_features`. Set `bias=true` for an additive bias vector
initialized to zero.
"""
function Linear(
    in_features::Integer, out_features::Integer; bias::Bool=false, init=Flux.glorot_uniform
)
    weight = init(out_features, in_features)
    b = bias ? zeros(Float32, out_features) : nothing
    return Linear(weight, b)
end

function (m::Linear{W,Nothing})(x::AbstractArray) where {W}
    sz = size(x)
    x_flat = reshape(x, sz[1], :)
    y_flat = m.weight * x_flat
    return reshape(y_flat, size(m.weight, 1), sz[2:end]...)
end

function (m::Linear{W,<:AbstractVector})(x::AbstractArray) where {W}
    sz = size(x)
    x_flat = reshape(x, sz[1], :)
    y_flat = m.weight * x_flat .+ m.bias
    return reshape(y_flat, size(m.weight, 1), sz[2:end]...)
end

Flux.@layer Linear
Flux.Optimisers.trainable(m::Linear{W,Nothing}) where {W} = (; weight=m.weight)
function Flux.Optimisers.trainable(m::Linear{W,<:AbstractVector}) where {W}
    return (; weight=m.weight, bias=m.bias)
end

"""
    SiLUGatedMLP{G, U, D}

SiLU-Gated Multi-Layer Perceptron (gated FFN) layer used in modern transformer decoders (e.g. Llama).
\$\\text{SiLUGatedMLP}(x) = (\\text{SiLU}(W_{gate} x) \\odot W_{up} x) W_{down}\$
"""
struct SiLUGatedMLP{G,U,D}
    gate_proj::G
    up_proj::U
    down_proj::D
end

"""
    SiLUGatedMLP(in_features::Integer, hidden_features::Integer; init = Flux.glorot_uniform)

Construct a `SiLUGatedMLP` layer with input size `in_features` and hidden size `hidden_features`.
"""
function SiLUGatedMLP(
    in_features::Integer, hidden_features::Integer; init=Flux.glorot_uniform
)
    gate_proj = Linear(in_features, hidden_features; init=init)
    up_proj = Linear(in_features, hidden_features; init=init)
    down_proj = Linear(hidden_features, in_features; init=init)
    return SiLUGatedMLP(gate_proj, up_proj, down_proj)
end

function (m::SiLUGatedMLP)(x::AbstractArray)
    g = NNlib.swish.(m.gate_proj(x))
    u = m.up_proj(x)
    return m.down_proj(g .* u)
end

Flux.@layer SiLUGatedMLP
function Flux.Optimisers.trainable(m::SiLUGatedMLP)
    (; gate_proj=m.gate_proj, up_proj=m.up_proj, down_proj=m.down_proj)
end

"""
    MoEMLP{G, E}

Mixture-of-Experts MLP block (Mixtral-style). A bias-less `gate`
projection scores each token over `num_experts` experts; the per-token
top-`top_k` experts are softmaxed and renormalized, and the result is
the weighted sum of those experts' outputs.

Each expert is a `SiLUGatedMLP` (SwiGLU) with the standard
`gate_proj` / `up_proj` / `down_proj` field names. The router output
is computed in the model's working dtype (typically Float32 in this
release).

The forward pass is correctness-first: it batches tokens per expert
(so each expert is called once with its assigned slice of tokens
rather than once per token), but doesn't yet pipeline router-to-expert
or fuse the expert matmuls. Phase 4 will revisit.
"""
struct MoEMLP{G,E}
    gate::G
    experts::E
    num_experts::Int
    top_k::Int
    function MoEMLP(
        gate::Linear, experts::AbstractVector, num_experts::Integer, top_k::Integer
    )
        return new{typeof(gate),typeof(experts)}(gate, experts, Int(num_experts), Int(top_k))
    end
end

"""
    MoEMLP(hidden::Integer, intermediate::Integer, num_experts::Integer, top_k::Integer; init = Flux.glorot_uniform)

Construct a fresh MoE block: a bias-less `Linear(hidden, num_experts)`
gate plus `num_experts` independent `SiLUGatedMLP(hidden, intermediate)`
experts.
"""
function MoEMLP(
    hidden::Integer,
    intermediate::Integer,
    num_experts::Integer,
    top_k::Integer;
    init=Flux.glorot_uniform,
)
    top_k > 0 || throw(ArgumentError("top_k must be > 0"))
    top_k <= num_experts || throw(
        ArgumentError("top_k ($(top_k)) cannot exceed num_experts ($(num_experts))"),
    )
    gate = Linear(hidden, num_experts; init=init)
    experts = [SiLUGatedMLP(hidden, intermediate; init=init) for _ in 1:num_experts]
    return MoEMLP(gate, experts, Int(num_experts), Int(top_k))
end

function (m::MoEMLP)(x::AbstractArray)
    # x: (hidden, seq, batch). Flatten the trailing dims to one "token" axis.
    hidden = size(x, 1)
    trailing = size(x)[2:end]
    n_tokens = prod(trailing)
    flat = reshape(x, hidden, n_tokens)

    # Router scores → softmax probabilities (n_experts, n_tokens).
    router_logits = m.gate(flat)
    probs = NNlib.softmax(router_logits; dims=1)

    # Per-token: pick top-K experts and renormalize their probabilities.
    n_experts = m.num_experts
    top_k = m.top_k
    T = eltype(probs)

    selected = Matrix{Int}(undef, top_k, n_tokens)
    weights = Matrix{T}(undef, top_k, n_tokens)
    for i in 1:n_tokens
        col = view(probs, :, i)
        topk_idx = partialsortperm(col, 1:top_k; rev=true)
        w = T[col[k] for k in topk_idx]
        w ./= sum(w)
        @inbounds for k in 1:top_k
            selected[k, i] = topk_idx[k]
            weights[k, i] = w[k]
        end
    end

    out = zeros(T, hidden, n_tokens)

    # For each expert: gather tokens that chose it, run, scatter back with weights.
    for expert_idx in 1:n_experts
        # Collect (token, slot) pairs where this expert was selected.
        token_idxs = Int[]
        slot_idxs = Int[]
        @inbounds for i in 1:n_tokens, k in 1:top_k
            if selected[k, i] == expert_idx
                push!(token_idxs, i)
                push!(slot_idxs, k)
            end
        end
        isempty(token_idxs) && continue

        # Run the expert on the gathered slice.
        input_subset = flat[:, token_idxs]
        expert_out = m.experts[expert_idx](input_subset)   # (hidden, n_assigned)

        # Scatter-add weighted outputs.
        @inbounds for j in eachindex(token_idxs)
            ti = token_idxs[j]
            w = weights[slot_idxs[j], ti]
            for h in 1:hidden
                out[h, ti] += w * expert_out[h, j]
            end
        end
    end

    return reshape(out, hidden, trailing...)
end

Flux.@layer MoEMLP
Flux.Optimisers.trainable(m::MoEMLP) = (; gate=m.gate, experts=m.experts)

"""
    GeluGatedMLP{G, U, D}

Gated FFN variant that uses GELU (tanh approximation) as the activation
instead of SiLU. Used by Gemma2. The tanh form matches HF's
`gelu_pytorch_tanh`:
`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
"""
struct GeluGatedMLP{G,U,D}
    gate_proj::G
    up_proj::U
    down_proj::D
end

"""
    GeluGatedMLP(in_features::Integer, hidden_features::Integer; init = Flux.glorot_uniform)

Construct a `GeluGatedMLP` layer with input size `in_features` and hidden
size `hidden_features`.
"""
function GeluGatedMLP(
    in_features::Integer, hidden_features::Integer; init=Flux.glorot_uniform
)
    gate_proj = Linear(in_features, hidden_features; init=init)
    up_proj = Linear(in_features, hidden_features; init=init)
    down_proj = Linear(hidden_features, in_features; init=init)
    return GeluGatedMLP(gate_proj, up_proj, down_proj)
end

@inline _gelu_tanh(x) =
    Float32(0.5) * x * (1 + tanh(sqrt(Float32(2 / pi)) * (x + Float32(0.044715) * x^3)))

# Exact GELU, via the NNlib implementation (uses erf under the hood).
# BERT, RoBERTa, and GPT-NeoX / Pythia all use this rather than the tanh
# approximation that GPT-2 uses.
@inline _gelu_exact(x) = NNlib.gelu(x)

function (m::GeluGatedMLP)(x::AbstractArray)
    g = _gelu_tanh.(m.gate_proj(x))
    u = m.up_proj(x)
    return m.down_proj(g .* u)
end

Flux.@layer GeluGatedMLP
function Flux.Optimisers.trainable(m::GeluGatedMLP)
    (; gate_proj=m.gate_proj, up_proj=m.up_proj, down_proj=m.down_proj)
end

"""
    GeluMLP{F, P}

Standard (non-gated) two-layer FFN: `c_proj(gelu(c_fc(x)))`. Both
linears carry bias.

`approx` selects the GELU form: `true` uses the tanh approximation
(what HF ships for GPT-2), `false` uses exact GELU via NNlib's
erf-based implementation (what BERT, RoBERTa, and GPT-NeoX/Pythia
ship). The choice is parity-critical — the two agree to ~1e-3 on
typical activations.
"""
struct GeluMLP{F,P}
    c_fc::F
    c_proj::P
    approx::Bool
end

"""
    GeluMLP(hidden::Integer, intermediate::Integer; approx=true, init = Flux.glorot_uniform)

Construct a `GeluMLP` with `Linear(hidden, intermediate; bias=true)` →
GELU → `Linear(intermediate, hidden; bias=true)`. `approx=true` (the
default, matching GPT-2) uses the tanh form; `approx=false` uses
exact GELU.
"""
function GeluMLP(
    hidden::Integer, intermediate::Integer; approx::Bool=true, init=Flux.glorot_uniform
)
    c_fc = Linear(hidden, intermediate; bias=true, init=init)
    c_proj = Linear(intermediate, hidden; bias=true, init=init)
    return GeluMLP(c_fc, c_proj, approx)
end

function (m::GeluMLP)(x::AbstractArray)
    h = m.c_fc(x)
    activated = m.approx ? _gelu_tanh.(h) : _gelu_exact.(h)
    return m.c_proj(activated)
end

Flux.@layer GeluMLP
Flux.Optimisers.trainable(m::GeluMLP) = (; c_fc=m.c_fc, c_proj=m.c_proj)

"""
    GQA{Q, K, V, O, R}

Grouped-Query Attention (GQA) layer. Partitions queries into groups sharing key/value heads.
Accepts an optional `KVCache` to update/retrieve past keys and values.

When `window_size` is set, attention is restricted to a sliding window of the
`window_size` most-recent positions (inclusive of the current query), matching
Mistral's sliding-window attention. `nothing` (the default) is plain causal.
"""
struct GQA{Q,K,V,O,R}
    wq::Q
    wk::K
    wv::V
    wo::O
    rope::R
    num_heads_q::Int
    num_heads_k::Int
    head_dim::Int
    window_size::Union{Nothing,Int}
    softcap::Union{Nothing,Float32}      # Attention-score softcap (Gemma2).
    query_scale::Union{Nothing,Float32}  # Override for the `sqrt(head_dim)` divisor.
    causal::Bool                         # false for BERT-style bidirectional attention.
end

"""
    GQA(hidden_dim::Integer, num_heads_q::Integer, num_heads_k::Integer, head_dim::Integer, rope; window_size=nothing, qkv_bias=false, init = Flux.glorot_uniform)

Construct a `GQA` layer with key/value caching and RoPE integration.

`rope` is either a `RoPE` instance (applied to Q and K before the score
matmul) or `nothing` for architectures with no rotary embeddings
(GPT-2, BERT-style attention with learned positions handled upstream).

`qkv_bias=true` adds an additive bias to the Q/K/V projections (matching
Qwen2/2.5 and other architectures that bias QKV but not the output
projection). `wo_bias=true` additionally biases the output projection
(GPT-2 does this; the Llama family doesn't). The two flags are
independent because Qwen2 wants `qkv_bias=true, wo_bias=false`.

`causal=false` (default `true`) disables the causal mask — every query
sees every key. Used by encoder-only models (BERT, RoBERTa) where
attention is bidirectional.
"""
function GQA(
    hidden_dim::Integer,
    num_heads_q::Integer,
    num_heads_k::Integer,
    head_dim::Integer,
    rope::Union{Nothing,RoPE};
    window_size::Union{Nothing,Integer}=nothing,
    qkv_bias::Bool=false,
    wo_bias::Bool=false,
    softcap::Union{Nothing,Real}=nothing,
    query_scale::Union{Nothing,Real}=nothing,
    causal::Bool=true,
    init=Flux.glorot_uniform,
)
    wq = Linear(hidden_dim, num_heads_q * head_dim; bias=qkv_bias, init=init)
    wk = Linear(hidden_dim, num_heads_k * head_dim; bias=qkv_bias, init=init)
    wv = Linear(hidden_dim, num_heads_k * head_dim; bias=qkv_bias, init=init)
    wo = Linear(num_heads_q * head_dim, hidden_dim; bias=wo_bias, init=init)
    win = isnothing(window_size) ? nothing : Int(window_size)
    sc = isnothing(softcap) ? nothing : Float32(softcap)
    qs = isnothing(query_scale) ? nothing : Float32(query_scale)
    return GQA(
        wq,
        wk,
        wv,
        wo,
        rope,
        Int(num_heads_q),
        Int(num_heads_k),
        Int(head_dim),
        win,
        sc,
        qs,
        causal,
    )
end

function repeat_kv(x::AbstractArray, group_size::Integer)
    if group_size == 1
        return x
    end
    return repeat(x; inner=(1, group_size, 1, 1))
end

function (m::GQA)(x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing)
    if !isnothing(cache) && isnothing(step)
        throw(
            ArgumentError(
                "GQA was passed a KV-cache without a `step`; pass both or neither."
            ),
        )
    end

    seq_len = size(x, 2)
    batch_size = size(x, 3)

    # Projections
    q = m.wq(x)
    k = m.wk(x)
    v = m.wv(x)

    # Reshape for multi-head attention
    q = reshape(q, m.head_dim, m.num_heads_q, seq_len, batch_size)
    k = reshape(k, m.head_dim, m.num_heads_k, seq_len, batch_size)
    v = reshape(v, m.head_dim, m.num_heads_k, seq_len, batch_size)

    # Compute position ids if not specified — 0-indexed to match HF's RoPE convention.
    # `step` is a 1-indexed Julia cache slot; the corresponding absolute position is `step - 1`.
    if isnothing(position_ids)
        start_pos = isnothing(step) ? 0 : step - 1
        position_ids = collect(start_pos:(start_pos + seq_len - 1))
    end

    # Apply RoPE (skipped when m.rope === nothing, e.g. GPT-2 / BERT where
    # positional information is already in the input embedding).
    if isnothing(m.rope)
        q_rotated = q
        k_rotated = k
    else
        q_rotated = m.rope(q, position_ids)
        k_rotated = m.rope(k, position_ids)
    end

    # Handle KV cache updates/retrievals (no permutedims — cache layout matches k_rotated).
    if !isnothing(cache)
        cache_slice = step:(step + seq_len - 1)
        cache.k[:, :, cache_slice, :] .= k_rotated
        cache.v[:, :, cache_slice, :] .= v
        k_full = cache.k[:, :, 1:(step + seq_len - 1), :]
        v_full = cache.v[:, :, 1:(step + seq_len - 1), :]
    else
        k_full = k_rotated
        v_full = v
    end

    seq_len_kv = size(k_full, 3)

    # GQA repetition
    group_size = m.num_heads_q ÷ m.num_heads_k
    k_rep = repeat_kv(k_full, group_size)
    v_rep = repeat_kv(v_full, group_size)

    # Scaled dot-product attention
    q_perm = permutedims(q_rotated, (3, 1, 2, 4))
    k_perm = permutedims(k_rep, (3, 1, 2, 4))

    q_flat = reshape(q_perm, seq_len, m.head_dim, m.num_heads_q * batch_size)
    k_flat = reshape(k_perm, seq_len_kv, m.head_dim, m.num_heads_q * batch_size)
    k_flat_t = permutedims(k_flat, (2, 1, 3))

    scale = isnothing(m.query_scale) ? sqrt(Float32(m.head_dim)) : m.query_scale
    scores = batched_mul(q_flat, k_flat_t) ./ scale

    # Gemma2 caps attention scores before mask + softmax: `cap * tanh(s / cap)`.
    if !isnothing(m.softcap)
        cap = m.softcap
        scores = cap .* tanh.(scores ./ cap)
    end

    # Mask construction. Key at array index `j` is at absolute position `j - 1`
    # (0-indexed); query `i` is at `position_ids[i]`. Mask iff:
    #   - causal:   key_pos > query_pos (when m.causal), or
    #   - sliding:  query_pos - key_pos >= window_size (when set).
    # The seq_len==1 fast path holds for plain causal (single query never sees a
    # future key), but a sliding window of size W still needs masking once the
    # KV cache exceeds W entries. With m.causal=false (BERT-style bidirectional
    # attention) neither predicate applies and we skip the mask entirely.
    has_window = !isnothing(m.window_size)
    needs_causal_mask = m.causal && seq_len > 1
    needs_window_mask = has_window && seq_len_kv > m.window_size
    needs_mask = needs_causal_mask || needs_window_mask
    if needs_mask
        mask = if has_window && m.causal
            w = m.window_size
            [
                (j - 1) > position_ids[i] || position_ids[i] - (j - 1) >= w for
                i in 1:seq_len, j in 1:seq_len_kv
            ]
        elseif has_window   # bidirectional + sliding (rare combo)
            w = m.window_size
            [abs(position_ids[i] - (j - 1)) >= w for i in 1:seq_len, j in 1:seq_len_kv]
        else                # causal only
            [(j - 1) > position_ids[i] for i in 1:seq_len, j in 1:seq_len_kv]
        end
        mask_val = mask .* Float32(-1e9)
        scores = scores .+ reshape(mask_val, seq_len, seq_len_kv, 1)
    end

    probs = NNlib.softmax(scores; dims=2)

    v_perm = permutedims(v_rep, (3, 1, 2, 4))
    v_flat = reshape(v_perm, seq_len_kv, m.head_dim, m.num_heads_q * batch_size)

    out_flat = batched_mul(probs, v_flat)

    out_perm = reshape(out_flat, seq_len, m.head_dim, m.num_heads_q, batch_size)
    out = permutedims(out_perm, (2, 3, 1, 4))

    out_collapsed = reshape(out, m.num_heads_q * m.head_dim, seq_len, batch_size)
    return m.wo(out_collapsed)
end

Flux.@layer GQA
Flux.Optimisers.trainable(m::GQA) = (; wq=m.wq, wk=m.wk, wv=m.wv, wo=m.wo)

end # module Layers
