using Test
using Random
using Flux
using Allspark.Models
using Allspark.Models: load_state_dict!, llava_state_dict_map, forward_from_features

# Stub vision tower: returns fixed patch features, ignoring pixel_values. The
# LLaVA composition is vision-tower-agnostic, so this exercises it end-to-end
# without a real encoder.
struct _StubTower{F}
    features::F
end
(t::_StubTower)(_pixel_values) = t.features

function _tiny_llava_llm(; vocab=20)
    cfg = LlamaConfig(;
        vocab_size=vocab,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rope_theta=10000.0,
        tie_word_embeddings=false,
    )
    return LlamaForCausalLM(cfg)
end

# Generic synthetic state dict from a model's own map: each source has the
# destination field's shape (reversed when the transform is :transpose).
function _synth_from_map(m, mapping)
    out = Dict{String,Array{Float32}}()
    for (k, (path, t)) in mapping
        dst = Allspark.Models._resolve(m, path)
        shp = t === :transpose ? reverse(size(dst)) : size(dst)
        out[k] = randn(Float32, shp...)
    end
    return out
end

const IMG = 19   # image placeholder token id (< vocab)

@testset "_splice_image replaces image positions in order" begin
    Random.seed!(0xF1)
    text = randn(Float32, 8, 5, 1)
    imgf = randn(Float32, 8, 3, 1)
    ids = reshape([2, IMG, IMG, IMG, 5], :, 1)
    merged = Allspark.Models._splice_image(text, imgf, ids, IMG)

    @test merged[:, 1, 1] == text[:, 1, 1]     # text positions untouched
    @test merged[:, 5, 1] == text[:, 5, 1]
    @test merged[:, 2, 1] == imgf[:, 1, 1]     # image positions filled in order
    @test merged[:, 3, 1] == imgf[:, 2, 1]
    @test merged[:, 4, 1] == imgf[:, 3, 1]
end

@testset "_splice_image rejects a patch-count mismatch" begin
    text = randn(Float32, 8, 5, 1)
    ids = reshape([2, IMG, IMG, IMG, 5], :, 1)
    @test_throws DimensionMismatch Allspark.Models._splice_image(
        text, randn(Float32, 8, 2, 1), ids, IMG
    )
end

@testset "projector: linear_1 → gelu → linear_2" begin
    Random.seed!(0xF2)
    proj = LlavaMultiModalProjector(6, 8)
    x = randn(Float32, 6, 3, 2)
    got = proj(x)
    @test size(got) == (8, 3, 2)
    h = Allspark.Layers._gelu_exact.(proj.linear_1(reshape(x, 6, 6)))
    @test got ≈ reshape(proj.linear_2(h), 8, 3, 2)
end

@testset "forward shape and feature entry point" begin
    Random.seed!(0xF3)
    lm = _tiny_llava_llm()
    feats = randn(Float32, 6, 3, 1)
    m = LlavaForConditionalGeneration(_StubTower(feats), lm, 6; image_token_index=IMG)
    ids = reshape([2, IMG, IMG, IMG, 5], :, 1)

    logits = m(nothing, ids)
    @test size(logits) == (20, 5, 1)
    @test forward_from_features(m, feats, ids) ≈ logits
end

@testset "image features flow through (causally)" begin
    Random.seed!(0xF4)
    lm = _tiny_llava_llm()
    m = LlavaForConditionalGeneration(_StubTower(zeros(Float32, 6, 3, 1)), lm, 6; image_token_index=IMG)
    ids = reshape([2, IMG, IMG, IMG, 5], :, 1)
    f1 = randn(Float32, 6, 3, 1)
    f2 = randn(Float32, 6, 3, 1)

    l1 = forward_from_features(m, f1, ids)
    l2 = forward_from_features(m, f2, ids)
    # Position 1 precedes every image token, so causal attention leaves it fixed.
    @test l1[:, 1, 1] ≈ l2[:, 1, 1]
    # The final position attends over the image tokens, so it changes.
    @test !(l1[:, end, 1] ≈ l2[:, end, 1])
end

@testset "load_state_dict! populates projector + language model" begin
    Random.seed!(0xF5)
    lm = _tiny_llava_llm()
    m = LlavaForConditionalGeneration(_StubTower(zeros(Float32, 6, 3, 1)), lm, 6; image_token_index=IMG)
    mp = llava_state_dict_map(m)
    sd = _synth_from_map(m, mp)
    load_state_dict!(m, sd)

    @test m.projector.linear_1.weight == sd["multi_modal_projector.linear_1.weight"]
    @test m.projector.linear_2.bias == sd["multi_modal_projector.linear_2.bias"]
    @test m.language_model.model.layers[1].self_attn.wq.weight ==
        sd["language_model.model.layers.0.self_attn.q_proj.weight"]
    @test m.language_model.lm_head.weight == sd["language_model.lm_head.weight"]
    # Language keys are re-rooted under language_model; vision-tower keys absent.
    @test haskey(mp, "language_model.model.norm.weight")
    @test !any(startswith(k, "vision_tower") for k in keys(mp))
end

@testset "generate_multimodal greedy-decodes from a multimodal prompt" begin
    Random.seed!(0xF6)
    lm = _tiny_llava_llm()
    m = LlavaForConditionalGeneration(_StubTower(randn(Float32, 6, 3, 1)), lm, 6; image_token_index=IMG)
    prompt = [2, IMG, IMG, IMG, 5]

    out = generate_multimodal(m, nothing, prompt; max_new_tokens=4)
    @test out isa Vector{Int}
    @test out[1:5] == prompt
    @test length(out) <= 5 + 4
    @test all(0 .<= out .< 20)
end
