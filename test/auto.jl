using Test
using Random
using JSON3
using HuggingFaceTransformers
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models: _resolve, _bert_flavor, _resolve_architecture

# Round-trip through JSON so the parsers see a real `JSON3.Object` with string
# keys, the way a downloaded config.json arrives, rather than a hand-built Dict.
_cfg(pairs::Pair...) = JSON3.read(JSON3.write(Dict{String,Any}(pairs...)))

@testset "read_config" begin
    dir = mktempdir()
    write(joinpath(dir, "config.json"), """{"model_type": "llama", "vocab_size": 7}""")

    @test read_config(dir).vocab_size == 7                        # directory
    @test read_config(joinpath(dir, "config.json")).vocab_size == 7  # file
    @test_throws ArgumentError read_config(mktempdir())
end

@testset "decoder configs derive head_dim and kv heads" begin
    # Both are optional in HF configs: `head_dim` falls back to hidden ÷ heads and
    # `num_key_value_heads` to MHA. Every decoder family shares this rule.
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512,
    ]

    cfg = config_from_json(LlamaConfig, _cfg(base...))
    @test cfg.head_dim == 8                       # 64 ÷ 8
    @test cfg.num_key_value_heads == 8            # defaults to MHA
    @test cfg.rope_theta == 500_000.0             # Llama-3 default
    @test isnothing(cfg.rope_scaling)

    stated = config_from_json(
        LlamaConfig, _cfg(base..., "head_dim" => 16, "num_key_value_heads" => 2)
    )
    @test stated.head_dim == 16                   # explicit value wins
    @test stated.num_key_value_heads == 2
end

@testset "Llama rope_scaling" begin
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512,
    ]

    llama3 = config_from_json(
        LlamaConfig,
        _cfg(
            base...,
            "rope_scaling" => Dict(
                "rope_type" => "llama3", "factor" => 32.0,
                "low_freq_factor" => 1.0, "high_freq_factor" => 4.0,
                "original_max_position_embeddings" => 8192,
            ),
        ),
    )
    @test llama3.rope_scaling isa LlamaRopeScaling
    @test llama3.rope_scaling.factor == 32.0
    @test llama3.rope_scaling.original_max_position_embeddings == 8192

    # `"rope_type": "default"` means no scaling at all.
    plain = config_from_json(
        LlamaConfig, _cfg(base..., "rope_scaling" => Dict("rope_type" => "default"))
    )
    @test isnothing(plain.rope_scaling)

    # Linear and dynamic scaling use a different formula, so refuse them by name
    # instead of reading the keys they happen to share.
    @test_throws ArgumentError config_from_json(
        LlamaConfig,
        _cfg(base..., "rope_scaling" => Dict("type" => "linear", "factor" => 8.0)),
    )
end

@testset "null and missing mean the same thing" begin
    # Mistral v0.3 ships `"sliding_window": null`, v0.2 ships a number.
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512,
    ]
    @test isnothing(config_from_json(MistralConfig, _cfg(base...)).sliding_window)
    @test isnothing(
        config_from_json(
            MistralConfig, _cfg(base..., "sliding_window" => nothing)
        ).sliding_window,
    )
    @test config_from_json(
        MistralConfig, _cfg(base..., "sliding_window" => 4096)
    ).sliding_window == 4096
end

@testset "Qwen gates sliding_window on use_sliding_window" begin
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512, "sliding_window" => 4096,
    ]
    # Qwen2.5 states a window but leaves the flag off, so the window is inert.
    @test isnothing(config_from_json(QwenConfig, _cfg(base...)).sliding_window)
    @test config_from_json(
        QwenConfig, _cfg(base..., "use_sliding_window" => true)
    ).sliding_window == 4096
end

@testset "Gemma optional softcaps" begin
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512,
    ]
    bare = config_from_json(GemmaConfig, _cfg(base...))
    @test isnothing(bare.attn_logit_softcapping)
    @test isnothing(bare.final_logit_softcapping)
    @test isnothing(bare.query_pre_attn_scalar)
    @test bare.tie_word_embeddings                      # Gemma ties by default

    full = config_from_json(
        GemmaConfig,
        _cfg(
            base...,
            "attn_logit_softcapping" => 50.0,
            "final_logit_softcapping" => 30.0,
            "query_pre_attn_scalar" => 224,
        ),
    )
    @test full.attn_logit_softcapping == 50.0
    @test full.final_logit_softcapping == 30.0
    @test full.query_pre_attn_scalar == 224
end

@testset "Phi-3 refuses unsupported context extension" begin
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 4096,
    ]
    @test config_from_json(Phi3Config, _cfg(base...)) isa Phi3Config
    # The 128k variants need longrope and partial rotary; loading them would give
    # silently wrong logits, so both must throw.
    @test_throws ArgumentError config_from_json(
        Phi3Config, _cfg(base..., "partial_rotary_factor" => 0.5)
    )
    @test_throws ArgumentError config_from_json(
        Phi3Config, _cfg(base..., "rope_scaling" => Dict("type" => "longrope")),
    )
end

@testset "GPT-2 legacy key names" begin
    cfg = config_from_json(
        GPT2Config,
        _cfg(
            "vocab_size" => 50257, "n_embd" => 768, "n_layer" => 12, "n_head" => 12,
            "n_positions" => 1024, "n_inner" => nothing,
            "layer_norm_epsilon" => 1.0e-5,
        ),
    )
    @test cfg.hidden_size == 768                  # n_embd
    @test cfg.num_hidden_layers == 12             # n_layer
    @test cfg.num_attention_heads == 12           # n_head
    @test cfg.max_position_embeddings == 1024     # n_positions
    @test cfg.intermediate_size == 3072           # null n_inner -> 4 * hidden
    @test cfg.layer_norm_eps == 1.0e-5            # layer_norm_epsilon
    @test cfg.tie_word_embeddings

    @test config_from_json(
        GPT2Config,
        _cfg(
            "vocab_size" => 50257, "n_embd" => 768, "n_layer" => 12, "n_head" => 12,
            "n_positions" => 1024, "n_inner" => 5000,
        ),
    ).intermediate_size == 5000                   # stated n_inner wins
end

@testset "NeoX rotary key names" begin
    cfg = config_from_json(
        NeoXConfig,
        _cfg(
            "vocab_size" => 50304, "hidden_size" => 512, "intermediate_size" => 2048,
            "num_hidden_layers" => 6, "num_attention_heads" => 8,
            "max_position_embeddings" => 2048, "rotary_pct" => 0.25,
            "rotary_emb_base" => 10_000.0,
        ),
    )
    @test cfg.partial_rotary_factor == 0.25       # rotary_pct
    @test cfg.rope_theta == 10_000.0              # rotary_emb_base
    @test cfg.head_dim == 64                      # 512 ÷ 8
end

@testset "Mixtral requires the expert counts" begin
    base = [
        "vocab_size" => 32, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512,
    ]
    cfg = config_from_json(
        MixtralConfig,
        _cfg(base..., "num_local_experts" => 8, "num_experts_per_tok" => 2),
    )
    @test cfg.num_local_experts == 8
    @test cfg.num_experts_per_tok == 2
    # Without them there is no sensible default, so fail loudly.
    @test_throws ArgumentError config_from_json(MixtralConfig, _cfg(base...))
end

@testset "BERT and RoBERTa flavors come from the checkpoint shape" begin
    base = [
        "vocab_size" => 30522, "hidden_size" => 64, "intermediate_size" => 128,
        "num_hidden_layers" => 2, "num_attention_heads" => 8,
        "max_position_embeddings" => 512,
    ]

    bert = config_from_json(
        BertConfig,
        _cfg(base..., "model_type" => "bert", "architectures" => ["BertForMaskedLM"]),
    )
    @test bert.hf_prefix == "bert"
    @test bert.head_prefix == "cls.predictions"
    @test bert.position_embedding_offset == 0
    @test bert.max_position_embeddings == 512     # no offset applied

    roberta = config_from_json(
        BertConfig,
        _cfg(
            base...,
            "model_type" => "roberta", "architectures" => ["RobertaForMaskedLM"],
            "max_position_embeddings" => 514, "type_vocab_size" => 1,
            "pad_token_id" => 1,
        ),
    )
    @test roberta.hf_prefix == "roberta"
    @test roberta.head_prefix == "lm_head"
    @test roberta.position_embedding_offset == 2
    @test roberta.max_position_embeddings == 512  # 514 - 2, RoBERTa's padding_idx + 1
    @test roberta.type_vocab_size == 1
    @test roberta.pad_token_id == 1

    # Sentence-embedding checkpoints (bge, e5) export the bare trunk, so the
    # weights carry no namespace prefix.
    trunk = config_from_json(
        BertConfig,
        _cfg(base..., "model_type" => "bert", "architectures" => ["BertModel"]),
    )
    @test trunk.hf_prefix == ""
    @test _bert_flavor(_cfg("model_type" => "roberta")).prefix == "roberta"
end

@testset "nomic accepts both key spellings" begin
    gpt2_style = config_from_json(
        NomicBertConfig,
        _cfg(
            "n_embd" => 768, "n_head" => 12, "n_layer" => 12, "n_inner" => 3072,
            "rotary_emb_base" => 1000.0, "layer_norm_epsilon" => 1.0e-12,
            "vocab_size" => 30528,
        ),
    )
    @test gpt2_style.hidden_size == 768
    @test gpt2_style.num_attention_heads == 12
    @test gpt2_style.num_hidden_layers == 12
    @test gpt2_style.intermediate_size == 3072
    @test gpt2_style.rope_theta == 1000.0
    @test gpt2_style.head_dim == 64

    bert_style = config_from_json(
        NomicBertConfig,
        _cfg(
            "hidden_size" => 768, "num_attention_heads" => 12,
            "num_hidden_layers" => 12, "intermediate_size" => 3072,
            "rope_parameters" => Dict("rope_theta" => 1000.0),
            "layer_norm_eps" => 1.0e-12, "vocab_size" => 30528,
        ),
    )
    @test bert_style.hidden_size == gpt2_style.hidden_size
    @test bert_style.rope_theta == 1000.0          # nested rope_parameters
end

@testset "vision configs" begin
    # ViT states its labels only through id2label.
    vit = config_from_json(
        ViTConfig,
        _cfg(
            "hidden_size" => 768, "num_hidden_layers" => 12,
            "num_attention_heads" => 12, "intermediate_size" => 3072,
            "architectures" => ["ViTForImageClassification"],
            "id2label" => Dict(string(i) => "class$(i)" for i in 0:4),
        ),
    )
    @test vit.num_labels == 5                      # length(id2label)
    @test vit.hf_prefix == "vit"

    trunk = config_from_json(
        ViTConfig,
        _cfg(
            "hidden_size" => 768, "num_hidden_layers" => 12,
            "num_attention_heads" => 12, "intermediate_size" => 3072,
            "architectures" => ["ViTModel"], "num_labels" => 3,
        ),
    )
    @test trunk.hf_prefix == ""                    # bare trunk export
    @test trunk.num_labels == 3

    dino = config_from_json(
        Dinov2Config,
        _cfg("architectures" => ["Dinov2Model"], "num_register_tokens" => 4),
    )
    @test dino.hf_prefix == ""
    @test dino.num_register_tokens == 4
    @test dino.patch_size == 14                    # DINOv2 default
    # The giant variant's SwiGLU FFN is not implemented.
    @test_throws ArgumentError config_from_json(
        Dinov2Config, _cfg("use_swiglu_ffn" => true)
    )
end

@testset "SigLIP reads nested tower configs" begin
    cfg = config_from_json(
        SiglipConfig,
        _cfg(
            "architectures" => ["SiglipModel"],
            "vision_config" => Dict("hidden_size" => 96, "patch_size" => 8),
            "text_config" => Dict("hidden_size" => 64, "vocab_size" => 128),
        ),
    )
    @test cfg.vision.hidden_size == 96
    @test cfg.vision.patch_size == 8
    @test cfg.text.hidden_size == 64
    @test cfg.text.vocab_size == 128
    @test cfg.text.projection_size == 64           # falls back to text hidden_size

    # Missing towers fall back to the published base defaults.
    defaults = config_from_json(SiglipConfig, _cfg("architectures" => ["SiglipModel"]))
    @test defaults.vision.hidden_size == 768
    @test defaults.text.max_position_embeddings == 64
end

@testset "Whisper mirrors HF naming" begin
    cfg = config_from_json(
        WhisperConfig,
        _cfg(
            "vocab_size" => 51865, "num_mel_bins" => 80, "d_model" => 512,
            "encoder_layers" => 6, "encoder_attention_heads" => 8,
            "encoder_ffn_dim" => 2048, "decoder_layers" => 6,
            "decoder_attention_heads" => 8, "decoder_ffn_dim" => 2048,
        ),
    )
    @test cfg.d_model == 512
    @test cfg.encoder_ffn_dim == 2048
    @test cfg.max_target_positions == 448          # default
    @test cfg.decoder_start_token_id == 50258
end

@testset "architecture resolution" begin
    @test _resolve_architecture(_cfg("architectures" => ["LlamaForCausalLM"])) ==
        "LlamaForCausalLM"
    # Several heads listed: take the first one we support.
    @test _resolve_architecture(
        _cfg("architectures" => ["SomeFutureHead", "GPT2LMHeadModel"])
    ) == "GPT2LMHeadModel"
    # No architectures at all, so fall back to model_type.
    @test _resolve_architecture(_cfg("model_type" => "mistral")) ==
        "MistralForCausalLM"

    # Unsupported checkpoints name what they were, rather than loading a wrong
    # near neighbour.
    err = try
        _resolve_architecture(_cfg("architectures" => ["T5ForConditionalGeneration"]))
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("T5ForConditionalGeneration", err.msg)
    @test occursin("LlamaForCausalLM", err.msg)    # lists the supported set
    @test_throws ArgumentError _resolve_architecture(_cfg("model_type" => "t5"))
end

@testset "every registry entry is loadable" begin
    # Guards against an entry naming a config or wrapper that does not exist, or a
    # config type with no `config_from_json` method.
    for (arch, (config_type, model_type)) in AUTO_ARCHITECTURES
        @testset "$(arch)" begin
            @test config_type isa Type
            @test !isempty(methods(config_from_json, (Type{config_type}, Any)))
            @test hasmethod(model_type, Tuple{config_type})
        end
    end
end

@testset "load_pretrained end to end from a fabricated checkpoint" begin
    # Invert a state-dict map to write HF-named safetensors for a model we built,
    # which gives a real checkpoint on disk with no download. Then load it back
    # through the auto path and check every parameter survived the round trip.
    Random.seed!(0xA07)
    cfg = LlamaConfig(;
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=64, rope_theta=500_000.0, tie_word_embeddings=false,
    )
    source = LlamaForCausalLM(cfg)

    tensors = Dict{String,AbstractArray}()
    for (hf_key, (path, transform)) in llama_state_dict_map(cfg)
        array = _resolve(source, path)
        # `:transpose` means the loader transposes on the way in, so undo it here.
        tensors[hf_key] = transform === :transpose ? permutedims(array, (2, 1)) : array
    end

    dir = mktempdir()
    Models._save_safetensors(joinpath(dir, "model.safetensors"), tensors)
    write(
        joinpath(dir, "config.json"),
        JSON3.write(
            Dict(
                "architectures" => ["LlamaForCausalLM"], "model_type" => "llama",
                "vocab_size" => 32, "hidden_size" => 16, "intermediate_size" => 32,
                "num_hidden_layers" => 2, "num_attention_heads" => 4,
                "num_key_value_heads" => 2, "head_dim" => 4,
                "max_position_embeddings" => 64, "rope_theta" => 500_000.0,
                "tie_word_embeddings" => false,
            ),
        ),
    )

    loaded = load_pretrained(dir)
    @test loaded isa LlamaForCausalLM
    @test loaded.config.hidden_size == 16
    @test loaded.config.num_key_value_heads == 2

    ids = reshape([1, 5, 9, 2], :, 1)
    @test loaded(ids) ≈ source(ids)                # same weights, same logits

    # The top-level entry point takes a directory as-is, with no download.
    @test HuggingFaceTransformers.load(dir)(ids) ≈ source(ids)

    # A checkpoint we cannot build must fail before touching the weights.
    bad = mktempdir()
    write(
        joinpath(bad, "config.json"),
        """{"architectures": ["MambaForCausalLM"], "model_type": "mamba"}""",
    )
    @test_throws ArgumentError load_pretrained(bad)
end
