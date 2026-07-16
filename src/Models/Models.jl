"""
    Allspark.Models

Concrete model architectures (`LlamaForCausalLM`, `BertModel`, ...) plus the
state-dict mapping tables that translate HuggingFace parameter keys
(`model.layers.0.self_attn.q_proj.weight`) into Julia layer addresses.
A new model adds: config struct, layer wiring, state-dict map. That's it.
"""
module Models

using Flux
using Functors
using BFloat16s: BFloat16
using SafeTensors: load_safetensors, load_sharded_safetensors
using ..Layers:
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
    sdpa,
    QuantizedInt8Matrix,
    KVCache,
    _gelu_exact
using ..Tokenizers: Tokenizer, encode

export load_weights, load_into!
export TokenEmbedding, DecoderLayer, DecoderModel, build_caches, load_state_dict!
export LlamaConfig, LlamaRopeScaling, LlamaForCausalLM, llama_state_dict_map
export MistralConfig, MistralForCausalLM, mistral_state_dict_map
export QwenConfig, QwenForCausalLM, qwen_state_dict_map
export GemmaConfig, GemmaDecoderLayer, GemmaModel, GemmaForCausalLM, gemma_state_dict_map
export Phi3Config, Phi3ForCausalLM, phi3_state_dict_map
export GPT2Config, GPT2Model, GPT2ForCausalLM, gpt2_state_dict_map
export NeoXConfig, NeoXDecoderLayer, NeoXForCausalLM, neox_state_dict_map
export BertConfig,
    BertEmbeddings,
    BertEncoderLayer,
    BertEncoder,
    BertModel,
    BertLMHead,
    BertForMaskedLM,
    bert_state_dict_map
export BertEmbeddingModel, embed, mean_pool, cls_pool, l2_normalize
export NomicBertConfig, NomicBertModel, nomic_state_dict_map
export ViTConfig, ViTModel, ViTForImageClassification, vit_state_dict_map
export SiglipConfig, SiglipVisionConfig, SiglipTextConfig
export SiglipVisionModel, SiglipTextModel, SiglipModel, siglip_state_dict_map
export Dinov2Config, Dinov2Model, dinov2_state_dict_map
export WhisperConfig, WhisperModel, whisper_state_dict_map, transcribe
export MixtralConfig, MixtralForCausalLM, mixtral_state_dict_map
export convert_eltype, fp16, bf16, fp32
export QuantizedInt8Matrix, quantize_int8

include("weights.jl")
include("state_dict.jl")
include("decoder.jl")
include("llama.jl")
include("mistral.jl")
include("qwen.jl")
include("gemma.jl")
include("phi3.jl")
include("gpt2.jl")
include("neox.jl")
include("bert.jl")
include("embedding.jl")
include("nomic.jl")
include("vit.jl")
include("siglip.jl")
include("dinov2.jl")
include("whisper.jl")
include("mixtral.jl")
include("dtype.jl")
include("quantize.jl")

end # module Models
