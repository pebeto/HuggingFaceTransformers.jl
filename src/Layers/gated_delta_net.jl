"""
    GatedDeltaNet

Gated DeltaNet linear attention (Yang et al., 2024), the recurrent layer Qwen3.5
uses for three in four of its blocks. Instead of attending over every past token
it keeps, per value head, a `(head_k_dim, head_v_dim)` state `S` that each token
first decays and then corrects toward its own key/value pair:

    S ← exp(g) S
    S ← S + k (β (v - Sᵀk))ᵀ
    o = Sᵀ q

`q`, `k` and `v` come from one projection passed through a short causal
depthwise convolution (`conv1d`, kernel `size(conv1d, 3)`) and SiLU; `q` and `k`
are L2-normalized and `q` scaled by `1/√head_k_dim`. The decay is
`g = -exp(A_log) softplus(a + dt_bias)` and the step size `β = sigmoid(b)`, from
their own small projections. The output is RMS-normalized per head, scaled by
`norm_weight`, and gated by `silu(z)` before `out_proj`.

Fields mirror HF's parameter names. With more value than key heads, consecutive
value heads share a key head, as `repeat_interleave` gives in HF.

Pass a [`DeltaNetCache`](@ref) as `cache` to carry the convolution window and the
state from one call to the next; `step == 1` starts both fresh. This runs the
recurrence token by token, which on a CPU is cheap at these state sizes, where HF
uses a chunked form for prompts: the two agree to float32 rounding. Inference
only: the recurrence updates its state in place and is not differentiable.
"""
struct GatedDeltaNet{L1,L2,L3,L4,L5,C,V,W}
    in_proj_qkv::L1
    in_proj_z::L2
    in_proj_b::L3
    in_proj_a::L4
    out_proj::L5
    conv1d::C            # (conv_dim, 1, kernel), as HF stores it
    dt_bias::V
    A_log::V
    norm_weight::W       # (head_v_dim,)
    num_k_heads::Int
    num_v_heads::Int
    head_k_dim::Int
    head_v_dim::Int
    eps::Float32
end

Flux.@layer GatedDeltaNet

"""
    GatedDeltaNet(hidden, num_k_heads, num_v_heads, head_k_dim, head_v_dim;
                  conv_kernel = 4, eps = 1e-6)

A freshly initialized layer; load real weights through a model's state-dict map.
"""
function GatedDeltaNet(
    hidden::Integer,
    num_k_heads::Integer,
    num_v_heads::Integer,
    head_k_dim::Integer,
    head_v_dim::Integer;
    conv_kernel::Integer=4,
    eps::Real=1.0e-6,
    init=Flux.glorot_uniform,
)
    num_v_heads % num_k_heads == 0 ||
        throw(ArgumentError("value heads ($(num_v_heads)) must be a multiple of key heads ($(num_k_heads))"))
    key_dim, value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
    conv_dim = 2key_dim + value_dim
    return GatedDeltaNet(
        Linear(hidden, conv_dim; init=init),
        Linear(hidden, value_dim; init=init),
        Linear(hidden, num_v_heads; init=init),
        Linear(hidden, num_v_heads; init=init),
        Linear(value_dim, hidden; init=init),
        init(conv_dim, 1, conv_kernel),
        ones(Float32, num_v_heads),
        log.(Float32.(range(1, 16; length=num_v_heads))),
        ones(Float32, head_v_dim),
        Int(num_k_heads), Int(num_v_heads), Int(head_k_dim), Int(head_v_dim), Float32(eps),
    )
end

"""
    DeltaNetCache(layer::GatedDeltaNet, batch_size)

What a [`GatedDeltaNet`](@ref) carries between calls: the last `kernel - 1`
inputs to its convolution and its recurrent state, per sequence. Unlike a
[`KVCache`](@ref) its size does not grow with the sequence.
"""
mutable struct DeltaNetCache{C,S}
    conv::C      # (conv_dim, kernel - 1, batch)
    state::S     # (head_k_dim, head_v_dim, num_v_heads, batch)
end

function DeltaNetCache(m::GatedDeltaNet, batch_size::Integer)
    conv_dim, kernel = size(m.conv1d, 1), size(m.conv1d, 3)
    return DeltaNetCache(
        zeros(Float32, conv_dim, kernel - 1, batch_size),
        zeros(Float32, m.head_k_dim, m.head_v_dim, m.num_v_heads, batch_size),
    )
end

function reset!(cache::DeltaNetCache)
    fill!(cache.conv, 0)
    fill!(cache.state, 0)
    return cache
end

_l2normalize(x, eps) = x ./ sqrt.(sum(abs2, x; dims=1) .+ eps)
_silu(x) = x .* NNlib.sigmoid.(x)

function (m::GatedDeltaNet)(
    x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing, padding_mask=nothing
)
    hidden, seq_len, batch = size(x)
    # Padding tokens enter as zeros, as HF zeroes them, so a left-padded row's
    # state stays empty until its real tokens arrive.
    if !isnothing(padding_mask)
        current = padding_mask[(end - seq_len + 1):end, :]
        x = x .* reshape(Float32.(current), 1, seq_len, batch)
    end
    fresh = isnothing(cache) || isnothing(step) || step == 1
    fresh && !isnothing(cache) && reset!(cache)

    # Causal depthwise convolution over the last `kernel` inputs.
    qkv = m.in_proj_qkv(x)                               # (conv_dim, seq, batch)
    kernel = size(m.conv1d, 3)
    previous = isnothing(cache) ? zeros(Float32, size(qkv, 1), kernel - 1, batch) : cache.conv
    window = cat(previous, qkv; dims=2)
    conv = sum(m.conv1d[:, 1, j] .* window[:, j:(j + seq_len - 1), :] for j in 1:kernel)
    isnothing(cache) || (cache.conv .= window[:, (end - kernel + 2):end, :])
    conv = _silu(conv)

    hk, hv, nk, nv = m.head_k_dim, m.head_v_dim, m.num_k_heads, m.num_v_heads
    key_dim = nk * hk
    q = _l2normalize(reshape(conv[1:key_dim, :, :], hk, nk, seq_len, batch), 1.0f-6) ./ sqrt(Float32(hk))
    k = _l2normalize(reshape(conv[(key_dim + 1):(2key_dim), :, :], hk, nk, seq_len, batch), 1.0f-6)
    v = reshape(conv[(2key_dim + 1):end, :, :], hv, nv, seq_len, batch)
    z = reshape(m.in_proj_z(x), hv, nv, seq_len, batch)
    beta = NNlib.sigmoid.(m.in_proj_b(x))                 # (nv, seq, batch)
    decay = exp.(-exp.(m.A_log) .* NNlib.softplus.(m.in_proj_a(x) .+ m.dt_bias))

    state = isnothing(cache) ? zeros(Float32, hk, hv, nv, batch) : cache.state
    out = Array{Float32}(undef, hv, nv, seq_len, batch)
    group = nv ÷ nk
    heads = vec(collect(Iterators.product(1:nv, 1:batch)))
    Threads.@threads for i in eachindex(heads)
        h, b = heads[i]
        kh = (h - 1) ÷ group + 1
        _delta_rule!(view(out, :, h, :, b), view(state, :, :, h, b), view(q, :, kh, :, b),
            view(k, :, kh, :, b), view(v, :, h, :, b), view(beta, h, :, b), view(decay, h, :, b))
    end

    # Per-head RMS norm, scaled and gated by silu(z).
    rms = sqrt.(sum(abs2, out; dims=1) ./ hv .+ m.eps)
    gated = (out ./ rms) .* m.norm_weight .* _silu(z)
    return m.out_proj(reshape(gated, hv * nv, seq_len, batch))
end

# One head, every token in order. Each column `j` of the state is decayed, read
# against the key, corrected by the delta and read against the query in a single
# pass.
function _delta_rule!(out, S, q, k, v, beta, decay)
    hk, hv = size(S)
    @inbounds for t in axes(q, 2)
        d, bt = decay[t], beta[t]
        for j in 1:hv
            recalled = 0.0f0
            @simd for i in 1:hk
                S[i, j] *= d
                recalled += S[i, j] * k[i, t]
            end
            delta = (v[j, t] - recalled) * bt
            read = 0.0f0
            @simd for i in 1:hk
                S[i, j] += k[i, t] * delta
                read += S[i, j] * q[i, t]
            end
            out[j, t] = read
        end
    end
    return out
end
