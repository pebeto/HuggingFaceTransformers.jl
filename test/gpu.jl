using Test
using Random
using Flux
using NNlib
using Functors: isleaf, fleaves
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Layers: Linear, checkpoint
using HuggingFaceTransformers.Generation

# Same assertions on every backend. `HFT_GPU_BACKEND` selects the device and
# defaults to the JLArrays reference backend, so this file runs in the default
# suite with no hardware. See test/gpu_backend.jl for the targets.
include("gpu_backend.jl")

function _tiny_lm(; seed=0x91)
    cfg = LlamaConfig(;
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    Random.seed!(seed)
    return LlamaForCausalLM(cfg), cfg
end

@testset "forward runs on-device and matches CPU" begin
    lm, _ = _tiny_lm()
    ids = reshape([1, 2, 3, 4], :, 1)
    cpu_logits = lm(ids)

    dev_logits = to_device(lm)(ids)
    @test dev_logits isa DeviceArray                 # no seam forced the result back to CPU
    @test dev_approx(dev_logits, cpu_logits)
end

@testset "KV-cache decode runs on-device and matches CPU" begin
    lm, _ = _tiny_lm()
    dmodel = to_device(lm)

    caches = build_caches(dmodel, 16, 1)
    @test caches[1].k isa DeviceArray                # caches allocated on the model's device

    ids = reshape([1, 2, 3], :, 1)
    dev_prefill = dmodel(ids; caches=caches, step=1)
    @test dev_prefill isa DeviceArray

    cpu_prefill = lm(ids; caches=build_caches(lm, 16, 1), step=1)
    @test dev_approx(dev_prefill, cpu_prefill)
end

@testset "greedy generate on-device equals CPU" begin
    lm, _ = _tiny_lm()
    ids = [1, 2, 3, 4]
    cpu_out = generate(lm, ids; max_new_tokens=6)
    dev_out = generate(to_device(lm), ids; max_new_tokens=6)
    @test dev_out == cpu_out                     # identical greedy tokens end-to-end
end

@testset "partial-rotary (NeoX) forward stays on-device" begin
    Random.seed!(0x92)
    cfg = NeoXConfig(;
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=4,
        partial_rotary_factor=0.5,               # rotary_dim = 2: partial-rotary path (Array→copy fix)
        max_position_embeddings=64,
        rope_theta=10000.0,
        layer_norm_eps=1.0e-5,
    )
    lm = NeoXForCausalLM(cfg)
    ids = reshape([1, 2, 3, 4], :, 1)
    cpu_logits = lm(ids)
    dev_logits = to_device(lm)(ids)
    @test dev_logits isa DeviceArray
    @test dev_approx(dev_logits, cpu_logits)
end

@testset "Mixtral (MoE) forward on-device matches CPU" begin
    Random.seed!(0x93)
    cfg = MixtralConfig(;
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=16, rope_theta=1_000_000.0, rms_norm_eps=1.0e-5,
        num_local_experts=4, num_experts_per_tok=2,
    )
    lm = MixtralForCausalLM(cfg)
    ids = reshape([1, 5, 9, 2], :, 1)
    cpu_logits = lm(ids)
    dev_logits = to_device(lm)(ids)
    @test dev_logits isa DeviceArray
    @test dev_approx(dev_logits, cpu_logits)
end

@testset "vision encoders (ViT, SigLIP, DINOv2) on-device match CPU" begin
    Random.seed!(0x94)
    px = randn(Float32, 3, 8, 8, 1)               # pixel tensor moves to the device too
    dpx = dev(px)

    vit = ViTForImageClassification(
        ViTConfig(; hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=16, image_size=8, patch_size=4, num_channels=3, num_labels=5),
    )
    dvit = to_device(vit)(dpx)
    @test dvit isa DeviceArray
    @test dev_approx(dvit, vit(px))

    dino = Dinov2Model(
        Dinov2Config(; hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=16, image_size=8, patch_size=4, num_channels=3,
            layer_norm_eps=1.0e-6, num_register_tokens=2),
    )
    ddino = to_device(dino)(dpx)
    @test ddino isa DeviceArray
    @test dev_approx(ddino, dino(px))

    siglip = SiglipModel(
        SiglipConfig(;
            vision=SiglipVisionConfig(; hidden_size=8, intermediate_size=16,
                num_hidden_layers=2, num_attention_heads=2, num_channels=3,
                image_size=8, patch_size=4, layer_norm_eps=1.0e-6),
            text=SiglipTextConfig(; vocab_size=20, hidden_size=8, intermediate_size=16,
                num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=6,
                projection_size=8, layer_norm_eps=1.0e-6),
        ),
    )
    ids = rand(0:19, 5, 2)
    dsig = to_device(siglip)(dpx, ids)
    @test dsig isa DeviceArray
    @test dev_approx(dsig, siglip(px, ids))
end

@testset "Whisper (encoder-decoder) on-device matches CPU" begin
    Random.seed!(0x95)
    cfg = WhisperConfig(;
        vocab_size=20, num_mel_bins=4, d_model=8, encoder_layers=2,
        encoder_attention_heads=2, encoder_ffn_dim=16, decoder_layers=2,
        decoder_attention_heads=2, decoder_ffn_dim=16, max_source_positions=6,
        max_target_positions=8,
    )
    m = WhisperModel(cfg)
    feats = randn(Float32, 4, 6, 1)
    ids = reshape([1, 2, 3, 4], :, 1)
    cpu_logits = m(feats, ids)
    dev_logits = to_device(m)(dev(feats), ids)
    @test dev_logits isa DeviceArray
    @test dev_approx(dev_logits, cpu_logits)
end

@testset "embedding models (BERT, nomic) on-device match CPU" begin
    Random.seed!(0x96)
    ids = rand(0:63, 5, 2)

    bert = BertEmbeddingModel(
        BertConfig(; vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=16,
            type_vocab_size=2, layer_norm_eps=1.0e-12);
        pooling=:mean,
    )
    dbert = to_device(bert)(ids)
    @test dbert isa DeviceArray
    @test dev_approx(dbert, bert(ids))

    nomic = BertEmbeddingModel(
        NomicBertConfig(; vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, head_dim=4, rope_theta=1000.0),
    )
    dnomic = to_device(nomic)(ids)
    @test dnomic isa DeviceArray
    @test dev_approx(dnomic, nomic(ids))
end

@testset "LoRA + checkpointed gradients on-device match CPU" begin
    # The inference testsets above only exercise the forward pass. This one covers
    # the backward: the `checkpoint` rrule (which recomputes through `rrule_via_ad`)
    # and the `@ignore_derivatives` seams in RoPE and `sdpa`, which is where a
    # host-array allocation would either error under `allowscalar(false)` or, worse,
    # silently drag gradients back to the CPU. Mirrors test/train.jl's setup.
    Random.seed!(0x98)
    vocab, seqlen = 16, 8
    cfg = LlamaConfig(;
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=seqlen,
        rope_theta=10000.0,
        tie_word_embeddings=false,
    )
    # `lora_wrap` zero-initializes lora_B (the PEFT convention), which makes every
    # d(loss)/d(lora_A) exactly zero. Perturb B so both matrices carry a real
    # gradient and the comparison covers the whole adapter.
    function wrap_lora(x)
        l = lora_wrap(x, 4, 8.0)
        randn!(l.lora_B)
        l.lora_B .*= 0.02f0
        return l
    end
    model = fmap(LlamaForCausalLM(cfg); exclude=(x -> x isa Linear || isleaf(x))) do x
        x isa Linear ? wrap_lora(x) : x
    end

    data = reshape(rand(0:(vocab - 1), seqlen), :, 1)
    targets = Float32.(Flux.onehotbatch(vec(data[2:end, :]), 0:(vocab - 1)))

    function loss(m, tgt)
        logits = checkpoint(m, data)
        pred = reshape(logits[:, 1:(end - 1), :], vocab, :)
        return Flux.logitcrossentropy(pred, tgt)
    end

    # Compare the gradient trees leaf by leaf. `fleaves` walks both in the same
    # order because the structures are identical, which covers every parameter that
    # received a gradient without going through `destructure` (that path trips over
    # LoRALinear's non-array `scaling` field).
    array_leaves(g) = filter(x -> x isa AbstractArray, fleaves(g))

    cpu_loss, cpu_g = Flux.withgradient(m -> loss(m, targets), model)
    dev_loss, dev_g = Flux.withgradient(
        m -> loss(m, dev(targets)), to_device(model)
    )

    cpu_leaves, dev_leaves = array_leaves(cpu_g[1]), array_leaves(dev_g[1])

    @test isapprox(dev_loss, cpu_loss; rtol=DEV_RTOL)
    @test !isempty(dev_leaves)
    @test length(dev_leaves) == length(cpu_leaves)
    @test all(l -> l isa DeviceArray, dev_leaves)        # no gradient fell back to host
    @test all(l -> any(!iszero, Array(l)), dev_leaves)   # and none silently zeroed out
    @test all(
        ((d, c),) -> isapprox(Array(d), c; rtol=GRAD_RTOL),
        zip(dev_leaves, cpu_leaves),
    )
end

@testset "LLaVA composition on-device matches CPU" begin
    Random.seed!(0x97)
    lm = LlamaForCausalLM(
        LlamaConfig(; vocab_size=20, hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=32, rope_theta=10000.0,
            tie_word_embeddings=false),
    )
    feats = randn(Float32, 6, 3, 1)               # vision patch features (vision_hidden, n_patches, batch)
    ids = reshape([2, 19, 19, 19, 5], :, 1)       # 19 = image placeholder, ×3 patches

    cpu_model = LlavaForConditionalGeneration(feats, lm, 6; image_token_index=19)
    cpu_logits = forward_from_features(cpu_model, feats, ids)

    dev_model = to_device(cpu_model)              # same weights, moved to device
    dev_logits = forward_from_features(dev_model, dev(feats), ids)
    @test dev_logits isa DeviceArray
    @test dev_approx(dev_logits, cpu_logits)
end
