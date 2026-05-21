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
using ..Layers: RMSNorm, RoPE, Linear, SiLUGatedMLP, GQA, KVCache

export load_weights
export LlamaConfig,
    LlamaRopeScaling,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    TokenEmbedding,
    build_caches

include("weights.jl")
include("llama.jl")

end # module Models
