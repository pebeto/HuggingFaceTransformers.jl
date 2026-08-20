"""
    read_config(path) -> JSON3.Object

Parse a HuggingFace `config.json`. `path` is either the file itself or a
directory containing it (a snapshot directory from
`HFHub.snapshot_download`).
"""
function read_config(path::AbstractString)
    file = isdir(path) ? joinpath(path, "config.json") : path
    isfile(file) || throw(ArgumentError("config.json not found at $(file)"))
    return JSON3.read(read(file, String))::JSON3.Object
end

# HF writes an absent value either by omitting the key or by setting it to null
# (`"sliding_window": null` ships on Mistral v0.3), so both collapse to `default`.
function _hf(raw, key::Symbol, default)
    haskey(raw, key) || return default
    value = raw[key]
    return isnothing(value) ? default : value
end

_hf_int(raw, key::Symbol, default::Integer) = Int(_hf(raw, key, default))
_hf_float(raw, key::Symbol, default::Real) = Float64(_hf(raw, key, default))
_hf_bool(raw, key::Symbol, default::Bool) = Bool(_hf(raw, key, default))
_hf_str(raw, key::Symbol, default::AbstractString) = String(_hf(raw, key, default))

function _hf_opt_int(raw, key::Symbol)
    value = _hf(raw, key, nothing)
    return isnothing(value) ? nothing : Int(value)
end

function _hf_opt_float(raw, key::Symbol)
    value = _hf(raw, key, nothing)
    return isnothing(value) ? nothing : Float64(value)
end

function _hf_req(raw, key::Symbol)
    value = _hf(raw, key, nothing)
    isnothing(value) &&
        throw(ArgumentError("config.json is missing the required key `$(key)`"))
    return value
end

_hf_req_int(raw, key::Symbol) = Int(_hf_req(raw, key))

# `architectures` is a list because a checkpoint can serve several heads; the
# first entry is the one the weights were saved for, which is what HF's
# `AutoModel` keys off too.
function _hf_architectures(raw)
    archs = _hf(raw, :architectures, nothing)
    isnothing(archs) && return String[]
    return String[String(a) for a in archs]
end

"""
    config_from_json(::Type{C}, raw) -> C

Build a config of type `C` from a parsed HuggingFace `config.json`.

One method per architecture, because this is the only place that knows HF's key
names. Renames (`n_embd`, `rotary_pct`), nullable keys, values derived from
others (`head_dim` from `hidden_size ÷ num_attention_heads`), and the
checkpoint-shape knobs that are not config keys at all (`hf_prefix`) are all
resolved here, so every other consumer sees a plain struct.

Unsupported variants raise rather than load silently wrong: Phi-3's `longrope`
scaling and DINOv2's SwiGLU FFN both error.
"""
function config_from_json end

# Every decoder family derives these two identically, and every checkpoint in the
# wild either states them or omits them in favour of these fallbacks.
_head_dim(raw, hidden::Int, heads::Int) = _hf_int(raw, :head_dim, hidden ÷ heads)
_kv_heads(raw, heads::Int) = _hf_int(raw, :num_key_value_heads, heads)

function _rope_scaling_from_json(raw)
    scaling = _hf(raw, :rope_scaling, nothing)
    isnothing(scaling) && return nothing

    # Llama 3.x ships `rope_type = "llama3"`. The linear and dynamic-NTK variants
    # carry different keys and a different formula, so refuse them by name rather
    # than half-reading them.
    rope_type = _hf_str(scaling, :rope_type, _hf_str(scaling, :type, "llama3"))
    rope_type in ("llama3", "default") || throw(
        ArgumentError(
            "unsupported rope_scaling type `$(rope_type)`; only Llama-3 NTK " *
            "scaling is implemented",
        ),
    )
    rope_type == "default" && return nothing

    return LlamaRopeScaling(;
        factor=Float64(_hf_req(scaling, :factor)),
        low_freq_factor=Float64(_hf_req(scaling, :low_freq_factor)),
        high_freq_factor=Float64(_hf_req(scaling, :high_freq_factor)),
        original_max_position_embeddings=Int(
            _hf_req(scaling, :original_max_position_embeddings)
        ),
    )
end

function config_from_json(::Type{LlamaConfig}, raw)
    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    return LlamaConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        num_key_value_heads=_kv_heads(raw, heads),
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        rope_theta=_hf_float(raw, :rope_theta, 500_000.0),
        rms_norm_eps=_hf_float(raw, :rms_norm_eps, 1.0e-5),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, false),
        rope_scaling=_rope_scaling_from_json(raw),
    )
end

function config_from_json(::Type{MistralConfig}, raw)
    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    return MistralConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        num_key_value_heads=_kv_heads(raw, heads),
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        rope_theta=_hf_float(raw, :rope_theta, 10_000.0),
        rms_norm_eps=_hf_float(raw, :rms_norm_eps, 1.0e-5),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, false),
        sliding_window=_hf_opt_int(raw, :sliding_window),
    )
end

function config_from_json(::Type{QwenConfig}, raw)
    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    # Qwen gates its window on a separate flag, so a stated `sliding_window` is
    # inert unless `use_sliding_window` is also set. The flag itself has no field.
    window =
        _hf_bool(raw, :use_sliding_window, false) ? _hf_opt_int(raw, :sliding_window) :
        nothing
    return QwenConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        num_key_value_heads=_kv_heads(raw, heads),
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        rope_theta=_hf_float(raw, :rope_theta, 1_000_000.0),
        rms_norm_eps=_hf_float(raw, :rms_norm_eps, 1.0e-6),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, false),
        sliding_window=window,
    )
end

function config_from_json(::Type{GemmaConfig}, raw)
    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    return GemmaConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        num_key_value_heads=_kv_heads(raw, heads),
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        rope_theta=_hf_float(raw, :rope_theta, 10_000.0),
        rms_norm_eps=_hf_float(raw, :rms_norm_eps, 1.0e-6),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, true),
        sliding_window=_hf_opt_int(raw, :sliding_window),
        attn_logit_softcapping=_hf_opt_float(raw, :attn_logit_softcapping),
        final_logit_softcapping=_hf_opt_float(raw, :final_logit_softcapping),
        query_pre_attn_scalar=_hf_opt_int(raw, :query_pre_attn_scalar),
    )
end

function config_from_json(::Type{Phi3Config}, raw)
    # Phi-3 mini/medium 4k are supported; the 128k variants extend context with
    # longrope and partial rotary, neither of which is implemented. Refuse them
    # here instead of producing drifted logits.
    partial = _hf_float(raw, :partial_rotary_factor, 1.0)
    partial == 1.0 || throw(
        ArgumentError(
            "Phi-3 partial_rotary_factor=$(partial) is not supported; only 1.0 is"
        ),
    )
    isnothing(_hf(raw, :rope_scaling, nothing)) ||
        throw(ArgumentError("Phi-3 longrope scaling is not supported"))

    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    return Phi3Config(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        num_key_value_heads=_kv_heads(raw, heads),
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        rope_theta=_hf_float(raw, :rope_theta, 10_000.0),
        rms_norm_eps=_hf_float(raw, :rms_norm_eps, 1.0e-5),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, false),
        sliding_window=_hf_opt_int(raw, :sliding_window),
    )
end

function config_from_json(::Type{MixtralConfig}, raw)
    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    return MixtralConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        num_key_value_heads=_kv_heads(raw, heads),
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        num_local_experts=_hf_req_int(raw, :num_local_experts),
        num_experts_per_tok=_hf_req_int(raw, :num_experts_per_tok),
        rope_theta=_hf_float(raw, :rope_theta, 1_000_000.0),
        rms_norm_eps=_hf_float(raw, :rms_norm_eps, 1.0e-5),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, false),
        sliding_window=_hf_opt_int(raw, :sliding_window),
    )
end

function config_from_json(::Type{GPT2Config}, raw)
    # GPT-2 predates HF's naming convention, so almost every key is renamed.
    hidden = _hf_req_int(raw, :n_embd)
    return GPT2Config(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_int(raw, :n_inner, 4 * hidden),
        num_hidden_layers=_hf_req_int(raw, :n_layer),
        num_attention_heads=_hf_req_int(raw, :n_head),
        max_position_embeddings=_hf_req_int(raw, :n_positions),
        layer_norm_eps=_hf_float(raw, :layer_norm_epsilon, 1.0e-5),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, true),
    )
end

function config_from_json(::Type{NeoXConfig}, raw)
    hidden = _hf_req_int(raw, :hidden_size)
    heads = _hf_req_int(raw, :num_attention_heads)
    return NeoXConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=hidden,
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=heads,
        head_dim=_head_dim(raw, hidden, heads),
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings),
        rope_theta=_hf_float(raw, :rotary_emb_base, 10_000.0),
        partial_rotary_factor=_hf_float(raw, :rotary_pct, 0.25),
        layer_norm_eps=_hf_float(raw, :layer_norm_eps, 1.0e-5),
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, false),
    )
end

# BERT and RoBERTa share this config, and the differences between them are
# checkpoint-shape facts rather than config keys: which namespace the weights
# live under, how the MLM head is named, and RoBERTa's `padding_idx + 1`
# position offset. Derive all three from `model_type` and `architectures`.
function _bert_flavor(raw)
    model_type = _hf_str(raw, :model_type, "bert")
    archs = _hf_architectures(raw)
    trunk_only = any(a -> a in ("BertModel", "RobertaModel"), archs)

    if model_type == "roberta" || any(a -> startswith(a, "Roberta"), archs)
        # Sentence-embedding checkpoints store the trunk at the top level.
        return (offset=2, prefix=trunk_only ? "" : "roberta", head="lm_head")
    end
    return (offset=0, prefix=trunk_only ? "" : "bert", head="cls.predictions")
end

function config_from_json(::Type{BertConfig}, raw)
    flavor = _bert_flavor(raw)
    return BertConfig(;
        vocab_size=_hf_req_int(raw, :vocab_size),
        hidden_size=_hf_req_int(raw, :hidden_size),
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=_hf_req_int(raw, :num_attention_heads),
        # RoBERTa reserves the first two position slots for padding, so the
        # usable table is shorter than the stated maximum.
        max_position_embeddings=_hf_req_int(raw, :max_position_embeddings) -
                                flavor.offset,
        type_vocab_size=_hf_int(raw, :type_vocab_size, 2),
        layer_norm_eps=_hf_float(raw, :layer_norm_eps, 1.0e-12),
        pad_token_id=_hf_int(raw, :pad_token_id, 0),
        position_embedding_offset=flavor.offset,
        hf_prefix=flavor.prefix,
        head_prefix=flavor.head,
        tie_word_embeddings=_hf_bool(raw, :tie_word_embeddings, true),
    )
end

function config_from_json(::Type{NomicBertConfig}, raw)
    # NomicBERT ships GPT-2-style key names, and newer revisions moved the rotary
    # base into a nested `rope_parameters`. Accept every spelling.
    hidden = _hf_int(raw, :n_embd, _hf_int(raw, :hidden_size, 768))
    heads = _hf_int(raw, :n_head, _hf_int(raw, :num_attention_heads, 12))
    rope_theta = if haskey(raw, :rotary_emb_base)
        _hf_float(raw, :rotary_emb_base, 1000.0)
    elseif haskey(raw, :rope_parameters)
        _hf_float(raw[:rope_parameters], :rope_theta, 1000.0)
    else
        1000.0
    end
    return NomicBertConfig(;
        vocab_size=_hf_int(raw, :vocab_size, 30528),
        hidden_size=hidden,
        intermediate_size=_hf_int(raw, :n_inner, _hf_int(raw, :intermediate_size, 4 * hidden)),
        num_hidden_layers=_hf_int(raw, :n_layer, _hf_int(raw, :num_hidden_layers, 12)),
        num_attention_heads=heads,
        head_dim=_hf_int(raw, :head_dim, hidden ÷ heads),
        type_vocab_size=_hf_int(raw, :type_vocab_size, 2),
        layer_norm_eps=_hf_float(
            raw, :layer_norm_epsilon, _hf_float(raw, :layer_norm_eps, 1.0e-12)
        ),
        rope_theta=rope_theta,
        pad_token_id=_hf_int(raw, :pad_token_id, 0),
    )
end

function config_from_json(::Type{ViTConfig}, raw)
    # Classification checkpoints namespace the trunk under `vit.`; a bare
    # `ViTModel` export does not.
    archs = _hf_architectures(raw)
    prefix = any(a -> a == "ViTModel", archs) ? "" : "vit"
    labels = if haskey(raw, :id2label)
        length(raw[:id2label])
    else
        _hf_int(raw, :num_labels, 1000)
    end
    return ViTConfig(;
        hidden_size=_hf_req_int(raw, :hidden_size),
        num_hidden_layers=_hf_req_int(raw, :num_hidden_layers),
        num_attention_heads=_hf_req_int(raw, :num_attention_heads),
        intermediate_size=_hf_req_int(raw, :intermediate_size),
        image_size=_hf_int(raw, :image_size, 224),
        patch_size=_hf_int(raw, :patch_size, 16),
        num_channels=_hf_int(raw, :num_channels, 3),
        layer_norm_eps=_hf_float(raw, :layer_norm_eps, 1.0e-12),
        num_labels=labels,
        qkv_bias=_hf_bool(raw, :qkv_bias, true),
        hf_prefix=prefix,
    )
end

function config_from_json(::Type{Dinov2Config}, raw)
    _hf_bool(raw, :use_swiglu_ffn, false) &&
        throw(ArgumentError("DINOv2 SwiGLU FFN (giant variant) is not supported yet"))
    archs = _hf_architectures(raw)
    prefix = any(a -> a == "Dinov2Model", archs) ? "" : "dinov2"
    return Dinov2Config(;
        hidden_size=_hf_int(raw, :hidden_size, 768),
        num_hidden_layers=_hf_int(raw, :num_hidden_layers, 12),
        num_attention_heads=_hf_int(raw, :num_attention_heads, 12),
        intermediate_size=_hf_int(raw, :intermediate_size, 3072),
        image_size=_hf_int(raw, :image_size, 518),
        patch_size=_hf_int(raw, :patch_size, 14),
        num_channels=_hf_int(raw, :num_channels, 3),
        layer_norm_eps=_hf_float(raw, :layer_norm_eps, 1.0e-6),
        layerscale_value=_hf_float(raw, :layerscale_value, 1.0),
        num_register_tokens=_hf_int(raw, :num_register_tokens, 0),
        qkv_bias=_hf_bool(raw, :qkv_bias, true),
        use_swiglu_ffn=false,
        hf_prefix=prefix,
    )
end

function config_from_json(::Type{SiglipVisionConfig}, raw)
    return SiglipVisionConfig(;
        hidden_size=_hf_int(raw, :hidden_size, 768),
        intermediate_size=_hf_int(raw, :intermediate_size, 3072),
        num_hidden_layers=_hf_int(raw, :num_hidden_layers, 12),
        num_attention_heads=_hf_int(raw, :num_attention_heads, 12),
        num_channels=_hf_int(raw, :num_channels, 3),
        image_size=_hf_int(raw, :image_size, 224),
        patch_size=_hf_int(raw, :patch_size, 16),
        layer_norm_eps=_hf_float(raw, :layer_norm_eps, 1.0e-6),
    )
end

function config_from_json(::Type{SiglipTextConfig}, raw)
    hidden = _hf_int(raw, :hidden_size, 768)
    return SiglipTextConfig(;
        vocab_size=_hf_int(raw, :vocab_size, 32000),
        hidden_size=hidden,
        intermediate_size=_hf_int(raw, :intermediate_size, 3072),
        num_hidden_layers=_hf_int(raw, :num_hidden_layers, 12),
        num_attention_heads=_hf_int(raw, :num_attention_heads, 12),
        max_position_embeddings=_hf_int(raw, :max_position_embeddings, 64),
        projection_size=_hf_int(raw, :projection_size, hidden),
        layer_norm_eps=_hf_float(raw, :layer_norm_eps, 1.0e-6),
    )
end

function config_from_json(::Type{SiglipConfig}, raw)
    # The towers live in nested objects, and the container field names differ
    # from HF's keys.
    vision = _hf(raw, :vision_config, nothing)
    text = _hf(raw, :text_config, nothing)
    return SiglipConfig(;
        vision=isnothing(vision) ? SiglipVisionConfig() :
               config_from_json(SiglipVisionConfig, vision),
        text=isnothing(text) ? SiglipTextConfig() :
             config_from_json(SiglipTextConfig, text),
    )
end

function config_from_json(::Type{WhisperConfig}, raw)
    # Whisper's own naming (`d_model`, `*_ffn_dim`) is mirrored verbatim.
    return WhisperConfig(;
        vocab_size=_hf_int(raw, :vocab_size, 51865),
        num_mel_bins=_hf_int(raw, :num_mel_bins, 80),
        d_model=_hf_int(raw, :d_model, 512),
        encoder_layers=_hf_int(raw, :encoder_layers, 6),
        encoder_attention_heads=_hf_int(raw, :encoder_attention_heads, 8),
        encoder_ffn_dim=_hf_int(raw, :encoder_ffn_dim, 2048),
        decoder_layers=_hf_int(raw, :decoder_layers, 6),
        decoder_attention_heads=_hf_int(raw, :decoder_attention_heads, 8),
        decoder_ffn_dim=_hf_int(raw, :decoder_ffn_dim, 2048),
        max_source_positions=_hf_int(raw, :max_source_positions, 1500),
        max_target_positions=_hf_int(raw, :max_target_positions, 448),
        decoder_start_token_id=_hf_int(raw, :decoder_start_token_id, 50258),
        eos_token_id=_hf_int(raw, :eos_token_id, 50257),
    )
end

"""
    AUTO_ARCHITECTURES

Maps the strings HuggingFace writes into `config.json`'s `architectures` array
onto the config type to parse and the wrapper type to build. This is the dispatch
table [`load_pretrained`](@ref) walks; adding a model means adding one entry.

Trunk-only entries (`BertModel`, `NomicBertModel`) build
[`BertEmbeddingModel`](@ref), since a bare encoder is what embedding checkpoints
ship and pooling is the only thing they add.
"""
const AUTO_ARCHITECTURES = Dict{String,Tuple{Any,Any}}(
    "LlamaForCausalLM" => (LlamaConfig, LlamaForCausalLM),
    "MistralForCausalLM" => (MistralConfig, MistralForCausalLM),
    "Qwen2ForCausalLM" => (QwenConfig, QwenForCausalLM),
    "Gemma2ForCausalLM" => (GemmaConfig, GemmaForCausalLM),
    "Phi3ForCausalLM" => (Phi3Config, Phi3ForCausalLM),
    "GPT2LMHeadModel" => (GPT2Config, GPT2ForCausalLM),
    "GPTNeoXForCausalLM" => (NeoXConfig, NeoXForCausalLM),
    "MixtralForCausalLM" => (MixtralConfig, MixtralForCausalLM),
    "BertForMaskedLM" => (BertConfig, BertForMaskedLM),
    "RobertaForMaskedLM" => (BertConfig, BertForMaskedLM),
    "BertModel" => (BertConfig, BertEmbeddingModel),
    "RobertaModel" => (BertConfig, BertEmbeddingModel),
    "NomicBertModel" => (NomicBertConfig, BertEmbeddingModel),
    "ViTForImageClassification" => (ViTConfig, ViTForImageClassification),
    "SiglipModel" => (SiglipConfig, SiglipModel),
    "Dinov2Model" => (Dinov2Config, Dinov2Model),
    "WhisperForConditionalGeneration" => (WhisperConfig, WhisperModel),
)

# Pick the architecture the weights were saved for. HF omits `architectures` on
# some community checkpoints, so fall back to the one entry a `model_type` can
# name unambiguously.
const _MODEL_TYPE_FALLBACK = Dict{String,String}(
    "llama" => "LlamaForCausalLM",
    "mistral" => "MistralForCausalLM",
    "qwen2" => "Qwen2ForCausalLM",
    "gemma2" => "Gemma2ForCausalLM",
    "phi3" => "Phi3ForCausalLM",
    "gpt2" => "GPT2LMHeadModel",
    "gpt_neox" => "GPTNeoXForCausalLM",
    "mixtral" => "MixtralForCausalLM",
    "nomic_bert" => "NomicBertModel",
    "siglip" => "SiglipModel",
    "dinov2" => "Dinov2Model",
    "whisper" => "WhisperForConditionalGeneration",
)

function _resolve_architecture(raw)
    for arch in _hf_architectures(raw)
        haskey(AUTO_ARCHITECTURES, arch) && return arch
    end

    model_type = _hf_str(raw, :model_type, "")
    haskey(_MODEL_TYPE_FALLBACK, model_type) && return _MODEL_TYPE_FALLBACK[model_type]

    stated = _hf_architectures(raw)
    described = isempty(stated) ? "model_type `$(model_type)`" :
                "architectures $(stated)"
    throw(
        ArgumentError(
            "unsupported checkpoint: $(described). Supported architectures are " *
            "$(sort(collect(keys(AUTO_ARCHITECTURES)))). Build the config and " *
            "model by hand for anything else.",
        ),
    )
end

"""
    load_pretrained(dir; kwargs...) -> model

Build a model from a local snapshot directory and load its weights. `dir` holds
a HuggingFace `config.json` plus safetensors (single or sharded).

The architecture comes from `config.json`, so no type needs naming:

```julia
dir = HFHub.snapshot_download("gpt2")
lm = load_pretrained(dir)
```

`kwargs` reach the model constructor, which is how embedding models take their
pooling: `load_pretrained(dir; pooling = :mean)` for E5, against the `:cls`
default that BGE wants.

Throws if the architecture is not in [`AUTO_ARCHITECTURES`](@ref) rather than
guessing a near neighbour. See [`HuggingFaceTransformers.load`](@ref) to download
and load in one call.
"""
function load_pretrained(dir::AbstractString; kwargs...)
    raw = read_config(dir)
    config_type, model_type = AUTO_ARCHITECTURES[_resolve_architecture(raw)]
    model = model_type(config_from_json(config_type, raw); kwargs...)
    return load_state_dict!(model, load_weights(dir))
end
