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

export KVCache, RMSNorm, RoPE, Linear, SiLUGatedMLP, GQA

mutable struct KVCache{T<:AbstractArray}
    k::T
    v::T
end

function KVCache(
    batch_size::Integer,
    n_kv_heads::Integer,
    max_seq::Integer,
    head_dim::Integer;
    eltype=Float32,
)
    k = zeros(eltype, batch_size, n_kv_heads, max_seq, head_dim)
    v = zeros(eltype, batch_size, n_kv_heads, max_seq, head_dim)
    return KVCache(k, v)
end

Flux.@layer KVCache
Flux.Optimisers.trainable(::KVCache) = (;)

struct RMSNorm{W,T}
    weight::W
    eps::T
end

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

struct RoPE{F}
    inv_freq::F
end

function RoPE(
    dim::Integer;
    base::Real=500000.0,
    scaling_factor::Union{Nothing,Real}=nothing,
    low_freq_factor::Union{Nothing,Real}=nothing,
    high_freq_factor::Union{Nothing,Real}=nothing,
    old_context_len::Union{Nothing,Integer}=nothing,
)
    half_dim = dim ÷ 2
    indices = 0:2:(dim - 2)
    inv_freq = 1.0 ./ (base .^ (indices ./ dim))

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

    return RoPE(Float32.(inv_freq))
end

function (m::RoPE)(x::AbstractArray, position_ids::AbstractVector)
    # Compute rotation angles: thetas of shape (half_dim, seq_len)
    thetas = m.inv_freq * reshape(position_ids, 1, :)
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

Flux.@layer RoPE
Flux.Optimisers.trainable(::RoPE) = (;)

struct Linear{W}
    weight::W
end

function Linear(in_features::Integer, out_features::Integer; init=Flux.glorot_uniform)
    weight = init(out_features, in_features)
    return Linear(weight)
end

function (m::Linear)(x::AbstractArray)
    sz = size(x)
    x_flat = reshape(x, sz[1], :)
    y_flat = m.weight * x_flat
    return reshape(y_flat, size(m.weight, 1), sz[2:end]...)
end

Flux.@layer Linear
Flux.Optimisers.trainable(m::Linear) = (; weight=m.weight)

struct SiLUGatedMLP{G,U,D}
    gate_proj::G
    up_proj::U
    down_proj::D
end

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

struct GQA{Q,K,V,O,R}
    wq::Q
    wk::K
    wv::V
    wo::O
    rope::R
    num_heads_q::Int
    num_heads_k::Int
    head_dim::Int
end

function GQA(
    hidden_dim::Integer,
    num_heads_q::Integer,
    num_heads_k::Integer,
    head_dim::Integer,
    rope::RoPE;
    init=Flux.glorot_uniform,
)
    wq = Linear(hidden_dim, num_heads_q * head_dim; init=init)
    wk = Linear(hidden_dim, num_heads_k * head_dim; init=init)
    wv = Linear(hidden_dim, num_heads_k * head_dim; init=init)
    wo = Linear(num_heads_q * head_dim, hidden_dim; init=init)
    return GQA(wq, wk, wv, wo, rope, Int(num_heads_q), Int(num_heads_k), Int(head_dim))
end

function repeat_kv(x::AbstractArray, group_size::Integer)
    if group_size == 1
        return x
    end
    return repeat(x; inner=(1, group_size, 1, 1))
end

function (m::GQA)(x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing)
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

    # Compute position ids if not specified
    if isnothing(position_ids)
        start_pos = isnothing(step) ? 1 : step
        position_ids = collect(start_pos:(start_pos + seq_len - 1))
    end

    # Apply RoPE
    q_rotated = m.rope(q, position_ids)
    k_rotated = m.rope(k, position_ids)

    # Handle KV cache updates/retrievals
    if !isnothing(cache) && !isnothing(step)
        k_perm = permutedims(k_rotated, (4, 2, 3, 1))
        v_perm = permutedims(v, (4, 2, 3, 1))

        cache.k[:, :, step:(step + seq_len - 1), :] .= k_perm
        cache.v[:, :, step:(step + seq_len - 1), :] .= v_perm

        k_full_perm = cache.k[:, :, 1:(step + seq_len - 1), :]
        v_full_perm = cache.v[:, :, 1:(step + seq_len - 1), :]

        k_full = permutedims(k_full_perm, (4, 2, 3, 1))
        v_full = permutedims(v_full_perm, (4, 2, 3, 1))
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

    scores = batched_mul(q_flat, k_flat_t) ./ sqrt(Float32(m.head_dim))

    # Causal masking
    if seq_len > 1
        mask = [j > position_ids[i] for i in 1:seq_len, j in 1:seq_len_kv]
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
