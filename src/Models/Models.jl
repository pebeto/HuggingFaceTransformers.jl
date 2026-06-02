"""
    Allspark.Models

Concrete model architectures (`LlamaForCausalLM`, `BertModel`, ...) plus the
state-dict mapping tables that translate HuggingFace parameter keys
(`model.layers.0.self_attn.q_proj.weight`) into Julia layer addresses.
A new model adds: config struct, layer wiring, state-dict map. That's it.
"""
module Models

using Flux
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
    GQA,
    KVCache

export load_weights, load_into!
export TokenEmbedding, DecoderLayer, DecoderModel, build_caches, load_state_dict!
export LlamaConfig, LlamaRopeScaling, LlamaForCausalLM, llama_state_dict_map
export MistralConfig, MistralForCausalLM, mistral_state_dict_map
export QwenConfig, QwenForCausalLM, qwen_state_dict_map
export GemmaConfig, GemmaDecoderLayer, GemmaModel, GemmaForCausalLM, gemma_state_dict_map
export Phi3Config, Phi3ForCausalLM, phi3_state_dict_map
export GPT2Config, GPT2Model, GPT2ForCausalLM, gpt2_state_dict_map
export NeoXConfig, NeoXDecoderLayer, NeoXForCausalLM, neox_state_dict_map

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

end # module Models
