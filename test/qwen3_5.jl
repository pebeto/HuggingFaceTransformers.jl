using Test
using Random
using JSON3
using NNlib
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models:
    Qwen3_5Config, Qwen3_5ForCausalLM, qwen3_5_state_dict_map, build_caches, _resolve
using HuggingFaceTransformers.Layers: GatedDeltaNet, DeltaNetCache, GQA, GemmaRMSNorm, RoPE
using HuggingFaceTransformers.Generation: generate, speculative_generate

# The delta rule written out the slow, obvious way, one token and head at a time:
# S ← exp(g) S; S ← S + k (β (v - Sᵀk))ᵀ; o = Sᵀq.
function _naive_delta_net(m::GatedDeltaNet, x::AbstractMatrix)
    T = size(x, 2)
    hk, hv, nk, nv = m.head_k_dim, m.head_v_dim, m.num_k_heads, m.num_v_heads
    qkv = m.in_proj_qkv.weight * x
    kernel = size(m.conv1d, 3)
    padded = hcat(zeros(Float32, size(qkv, 1), kernel - 1), qkv)
    conv = [sum(m.conv1d[c, 1, j] * padded[c, t + j - 1] for j in 1:kernel) for c in axes(qkv, 1), t in 1:T]
    conv = conv .* NNlib.sigmoid.(conv)
    z = m.in_proj_z.weight * x
    b = NNlib.sigmoid.(m.in_proj_b.weight * x)
    a = m.in_proj_a.weight * x
    out = zeros(Float32, hv * nv, T)
    for h in 1:nv
        kh = (h - 1) ÷ (nv ÷ nk)
        S = zeros(Float32, hk, hv)
        for t in 1:T
            q = conv[kh * hk .+ (1:hk), t]
            k = conv[nk * hk + kh * hk .+ (1:hk), t]
            v = conv[2nk * hk + (h - 1) * hv .+ (1:hv), t]
            q = q ./ sqrt(sum(abs2, q) + 1.0f-6) ./ sqrt(Float32(hk))
            k = k ./ sqrt(sum(abs2, k) + 1.0f-6)
            g = -exp(m.A_log[h]) * log1p(exp(a[h, t] + m.dt_bias[h]))
            S = exp(g) * S
            S = S + k * (b[h, t] * (v - S' * k))'
            o = S' * q
            gate = z[(h - 1) * hv .+ (1:hv), t]
            o = o ./ sqrt(sum(abs2, o) / hv + m.eps) .* m.norm_weight .* (gate .* NNlib.sigmoid.(gate))
            out[(h - 1) * hv .+ (1:hv), t] = o
        end
    end
    return m.out_proj.weight * out
end

function _random_delta_net(rng)
    m = GatedDeltaNet(16, 2, 4, 8, 8; conv_kernel=4)       # two value heads per key head
    m.conv1d .= randn(rng, Float32, size(m.conv1d)) .* 0.5f0
    m.dt_bias .= randn(rng, Float32, 4)
    m.A_log .= randn(rng, Float32, 4) .* 0.5f0
    m.norm_weight .= 1 .+ randn(rng, Float32, 8) .* 0.1f0
    return m
end

@testset verbose = true "GatedDeltaNet" begin
    rng = Xoshiro(11)
    m = _random_delta_net(rng)
    x = randn(rng, Float32, 16, 9, 1)
    full = m(x)

    @test full[:, :, 1] ≈ _naive_delta_net(m, x[:, :, 1]) rtol = 1.0e-4

    @testset "token by token through a cache equals one pass" begin
        cache = DeltaNetCache(m, 1)
        steps = [m(x[:, t:t, :]; cache=cache, step=t) for t in 1:9]
        @test reduce(hcat, (s[:, :, 1] for s in steps)) ≈ full[:, :, 1] rtol = 1.0e-5
        # Prompt, then single tokens.
        cache2 = DeltaNetCache(m, 1)
        prefix = m(x[:, 1:5, :]; cache=cache2, step=1)
        rest = [m(x[:, t:t, :]; cache=cache2, step=t) for t in 6:9]
        @test hcat(prefix[:, :, 1], (r[:, :, 1] for r in rest)...) ≈ full[:, :, 1] rtol = 1.0e-5
        # Step 1 starts over, whatever the cache held.
        @test m(x; cache=cache2, step=1) ≈ full
    end

    @testset "left padding leaves the real tokens untouched" begin
        padded = cat(randn(rng, Float32, 16, 3, 1), x; dims=2)
        mask = vcat(falses(3), trues(9))
        out = m(padded; padding_mask=reshape(mask, :, 1))
        @test out[:, 4:end, 1] ≈ full[:, :, 1] rtol = 1.0e-5
    end

    @test_throws ArgumentError GatedDeltaNet(16, 3, 4, 8, 8)
end

@testset "GQA output gate and zero-centered q/k norms" begin
    rng = Xoshiro(12)
    plain = GQA(16, 4, 2, 4, RoPE(4; base=10_000.0))
    gated = GQA(16, 4, 2, 4, RoPE(4; base=10_000.0); output_gate=true, qk_norm_eps=1.0e-6,
        qk_norm_zero_centered=true)
    @test gated.q_norm isa GemmaRMSNorm && size(gated.wq.weight) == (32, 16)
    # Per head, queries then gate rows. Zero gate rows make every gate sigmoid(0).
    for h in 1:4
        gated.wq.weight[(8h - 7):(8h - 4), :] .= plain.wq.weight[(4h - 3):(4h), :]
        gated.wq.weight[(8h - 3):(8h), :] .= 0
    end
    gated.wk.weight .= plain.wk.weight
    gated.wv.weight .= plain.wv.weight
    gated.wo.weight .= plain.wo.weight
    # Zero-centered norms start as identity scaling; plain has none, so compare
    # against a plain layer with equivalent RMSNorms instead of none at all.
    reference = GQA(16, 4, 2, 4, RoPE(4; base=10_000.0); qk_norm_eps=1.0e-6)
    for f in (:wq, :wk, :wv, :wo)
        getfield(reference, f).weight .= getfield(plain, f).weight
    end
    x = randn(rng, Float32, 16, 5, 1)
    @test gated(x) ≈ 0.5f0 .* reference(x)
end

function _tiny_qwen3_5_config(; kwargs...)
    return Qwen3_5Config(;
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=64, partial_rotary_factor=0.5, kwargs...,
    )
end

@testset verbose = true "Qwen3_5ForCausalLM" begin
    Random.seed!(0x35)
    cfg = _tiny_qwen3_5_config()
    lm = Qwen3_5ForCausalLM(cfg)
    @test [typeof(l.self_attn) <: GatedDeltaNet for l in lm.model.layers] == [true, true, true, false]
    @test lm.model.layers[4].self_attn.rope.rotary_dim == 4

    ids = reshape(rand(Xoshiro(13), 0:63, 9), :, 1)
    logits = lm(ids)
    @test size(logits) == (64, 9, 1) && all(isfinite, logits)

    @testset "cached decoding matches the full pass" begin
        caches = build_caches(lm, 32, 1)
        @test count(c -> c isa DeltaNetCache, caches) == 3
        lm(ids[1:6, :]; caches=caches, step=1)
        steps = [lm(ids[t:t, :]; caches=caches, step=t) for t in 7:9]
        @test all(isapprox(steps[j], logits[:, 6 + j:6 + j, :]; atol=1.0e-4) for j in 1:3)
    end

    @testset "batched generation with left padding matches one at a time" begin
        prompts = [[3, 7, 11, 2, 9], [5, 1]]
        batched = generate(lm, prompts; max_new_tokens=6)
        @test batched == [generate(lm, p; max_new_tokens=6) for p in prompts]
    end

    @testset "every mapped weight loads" begin
        sd = Dict{String,Array{Float32}}()
        for (key, (path, transform)) in qwen3_5_state_dict_map(cfg)
            a = _resolve(lm, path)
            sd[key] = transform === :transpose ? permutedims(a, (2, 1)) : copy(a)
        end
        @test haskey(sd, "model.language_model.layers.0.linear_attn.conv1d.weight")
        @test haskey(sd, "model.language_model.layers.3.self_attn.q_norm.weight")
        fresh = load_state_dict!(Qwen3_5ForCausalLM(cfg), sd)
        @test fresh(ids) ≈ logits
    end

    @test_throws ArgumentError speculative_generate(lm, lm, [1, 2, 3]; max_new_tokens=2)
end

@testset "config_from_json(Qwen3_5Config)" begin
    text = Dict(
        "vocab_size" => 64, "hidden_size" => 32, "intermediate_size" => 48,
        "num_hidden_layers" => 4, "num_attention_heads" => 4, "num_key_value_heads" => 2,
        "head_dim" => 8, "linear_num_key_heads" => 2, "linear_num_value_heads" => 4,
        "linear_key_head_dim" => 8, "linear_value_head_dim" => 8,
        "max_position_embeddings" => 64, "tie_word_embeddings" => true,
        "rope_parameters" => Dict("rope_type" => "default", "rope_theta" => 1.0e7,
            "partial_rotary_factor" => 0.25, "mrope_interleaved" => true),
    )
    parse(raw) = config_from_json(Qwen3_5Config, JSON3.read(JSON3.write(raw)))

    # Qwen's checkpoints are multimodal: the language model's settings are nested.
    nested = parse(Dict("architectures" => ["Qwen3_5ForConditionalGeneration"], "text_config" => text))
    @test nested.hf_prefix == "model.language_model" && nested.tie_word_embeddings
    @test nested.layer_types == [:linear_attention, :linear_attention, :linear_attention, :full_attention]
    @test nested.rope_theta == 1.0e7 && nested.partial_rotary_factor == 0.25

    flat = parse(merge(text, Dict("model_type" => "qwen3_5_text")))
    @test flat.hf_prefix == "model"

    @test_throws ArgumentError parse(merge(text, Dict("mlp_only_layers" => [1])))
    yarn = Dict("rope_type" => "yarn", "factor" => 4.0)
    @test_throws ArgumentError parse(merge(text, Dict("rope_parameters" => yarn)))
end
