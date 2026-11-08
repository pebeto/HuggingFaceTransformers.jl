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

    conv = _causal_conv_silu(m.conv1d, m.in_proj_qkv(x), isnothing(cache) ? nothing : cache.conv)
    z = m.in_proj_z(x)                                    # (value_dim, seq, batch)
    beta = NNlib.sigmoid.(m.in_proj_b(x))                 # (nv, seq, batch)
    decay = exp.(-exp.(m.A_log) .* NNlib.softplus.(m.in_proj_a(x) .+ m.dt_bias))

    hk, hv, nv = m.head_k_dim, m.head_v_dim, m.num_v_heads
    state = isnothing(cache) ? zeros(Float32, hk, hv, nv, batch) : cache.state
    out = Array{Float32}(undef, hv, nv, seq_len, batch)
    Threads.@threads for i in 1:(nv * batch)
        _delta_head!(out, state, conv, z, beta, decay, m, (i - 1) % nv + 1, (i - 1) ÷ nv + 1)
    end
    return m.out_proj(reshape(out, hv * nv, seq_len, batch))
end

# The causal depthwise convolution over the last `kernel` inputs, then SiLU. With a
# cache, `previous` holds the inputs before this call's first and is moved along.
function _causal_conv_silu(w, qkv, previous)
    conv_dim, seq_len, batch = size(qkv)
    kernel = size(w, 3)
    out = zeros(Float32, conv_dim, seq_len, batch)
    @inbounds for b in 1:batch, t in 1:seq_len, j in 1:kernel
        p = t + j - kernel                  # input position; ≤ 0 is before this call
        p < 1 && isnothing(previous) && continue
        @simd for c in 1:conv_dim
            input = p >= 1 ? qkv[c, p, b] : previous[c, p + kernel - 1, b]
            out[c, t, b] = muladd(w[c, 1, j], input, out[c, t, b])
        end
    end
    @inbounds @simd for i in eachindex(out)
        out[i] *= NNlib.sigmoid(out[i])
    end
    if !isnothing(previous)
        # The window's last kernel - 1 columns; each read is ahead of every write.
        @inbounds for i in 1:(kernel - 1), b in 1:batch
            p = seq_len + i - (kernel - 1)
            @simd for c in 1:conv_dim
                previous[c, i, b] = p >= 1 ? qkv[c, p, b] : previous[c, p + kernel - 1, b]
            end
        end
    end
    return out
end

# Value head h of sequence b, every token in order. q and k are L2-normalized on
# the fly (their scale factors folded into the sums), each column `j` of the state
# is decayed, read against the key, corrected by the delta and read against the
# query in a single pass, and the head's output is RMS-normalized and gated by
# silu(z).
function _delta_head!(out, S, conv, z, beta, decay, m::GatedDeltaNet, h, b)
    hk, hv, nk, nv = m.head_k_dim, m.head_v_dim, m.num_k_heads, m.num_v_heads
    kh = (h - 1) ÷ (nv ÷ nk)
    q0, k0, v0 = kh * hk, (nk + kh) * hk, 2nk * hk + (h - 1) * hv
    root = sqrt(Float32(hk))
    @inbounds for t in axes(conv, 2)
        sq = sk = 0.0f0
        @simd for i in 1:hk
            sq = muladd(conv[q0 + i, t, b], conv[q0 + i, t, b], sq)
            sk = muladd(conv[k0 + i, t, b], conv[k0 + i, t, b], sk)
        end
        qn, kn = 1 / (sqrt(sq + 1.0f-6) * root), 1 / sqrt(sk + 1.0f-6)
        d, bt = decay[h, t, b], beta[h, t, b]
        energy = 0.0f0
        for j in 1:hv
            recalled = 0.0f0
            @simd for i in 1:hk
                S[i, j, h, b] *= d
                recalled = muladd(S[i, j, h, b], conv[k0 + i, t, b], recalled)
            end
            delta = (conv[v0 + j, t, b] - recalled * kn) * bt * kn
            read = 0.0f0
            @simd for i in 1:hk
                S[i, j, h, b] = muladd(conv[k0 + i, t, b], delta, S[i, j, h, b])
                read = muladd(S[i, j, h, b], conv[q0 + i, t, b], read)
            end
            o = read * qn
            out[j, h, t, b] = o
            energy = muladd(o, o, energy)
        end
        inv_rms = 1 / sqrt(energy / hv + m.eps)
        @simd for j in 1:hv
            g = z[(h - 1) * hv + j, t, b]
            out[j, h, t, b] *= inv_rms * m.norm_weight[j] * (g * NNlib.sigmoid(g))
        end
    end
    return out
end
