```@meta
CurrentModule = HuggingFaceTransformers.Models
```

# Decoder language models

Causal LMs and their configs. Each `*_state_dict_map` returns the pure-data table that maps HF parameter names onto the Julia layer tree.

```@docs
GPT2Config
GPT2ForCausalLM
GPT2Model
GemmaConfig
GemmaDecoderLayer
GemmaForCausalLM
GemmaModel
LlamaConfig
LlamaForCausalLM
LlamaRopeScaling
MistralConfig
MistralForCausalLM
MixtralConfig
MixtralForCausalLM
NeoXConfig
NeoXDecoderLayer
NeoXForCausalLM
Phi3Config
Phi3ForCausalLM
QwenConfig
QwenForCausalLM
gemma_state_dict_map
gpt2_state_dict_map
llama_state_dict_map
mistral_state_dict_map
mixtral_state_dict_map
neox_state_dict_map
phi3_state_dict_map
qwen_state_dict_map
```
