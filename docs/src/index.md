```@meta
CurrentModule = HuggingFaceTransformers
```

# HuggingFaceTransformers.jl

Load and run HuggingFace transformer checkpoints in Julia. Weights come from
the Hub in safetensors format, tokenizers from `tokenizer.json`, and the forward
passes are plain Flux layers, so they differentiate and move to a GPU like any
other Flux model.

The spiritual successor to
[`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl), rebuilt
with a leaner dependency tree and a focus on decoder LLMs.

Every architecture except LLaVA has a numeric-parity test that compares logits
against a reference recorded from `transformers` in Python.

```@docs
HuggingFaceTransformers
```

## Installation

Julia 1.10 or newer. Not registered yet, so install from the repository:

```julia
julia> ]
pkg> dev https://github.com/pebeto/HuggingFaceTransformers.jl
```

## Quick start

GPT-2 needs no access token, which makes it the shortest end-to-end example.
Roughly 0.5 GB downloads to `~/.cache/huggingface/hub` on first run.

```julia
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer
using HuggingFaceTransformers.Models:
    GPT2Config, GPT2ForCausalLM, load_weights, load_state_dict!
using HuggingFaceTransformers.Generation: generate
using JSON3

dir = snapshot_download("gpt2")
raw = JSON3.read(read(joinpath(dir, "config.json"), String))

cfg = GPT2Config(;
    vocab_size=Int(raw.vocab_size),
    hidden_size=Int(raw.n_embd),
    intermediate_size=4 * Int(raw.n_embd),
    num_hidden_layers=Int(raw.n_layer),
    num_attention_heads=Int(raw.n_head),
    max_position_embeddings=Int(raw.n_positions),
)

lm = GPT2ForCausalLM(cfg)
load_state_dict!(lm, load_weights(dir))

tokenizer = load_tokenizer(dir)
println(generate(lm, tokenizer, "The Julia language is"; max_new_tokens=32))
```

Gated repositories (Llama and Gemma among them) need a token in `HF_TOKEN`, or
a `huggingface-cli login` done once. The package reads the same
`~/.cache/huggingface/token` file the Python library writes.

## Supported architectures

### Decoder language models

| Family | Type | Parity recorded against |
| --- | --- | --- |
| Llama 3.x | [`LlamaForCausalLM`](@ref) | `Llama-3.2-1B/3B-Instruct`, `Llama-3.1-8B/70B-Instruct` |
| Mistral | [`MistralForCausalLM`](@ref) | `Mistral-7B-Instruct-v0.2`, `v0.3` |
| Qwen2 / Qwen2.5 | [`QwenForCausalLM`](@ref) | `Qwen2.5-0.5B/1.5B/3B/7B-Instruct` |
| Gemma 2 | [`GemmaForCausalLM`](@ref) | `gemma-2-2b/9b/27b-it` |
| Phi-3 | [`Phi3ForCausalLM`](@ref) | `Phi-3-mini-4k-instruct`, `Phi-3-medium-4k-instruct` |
| GPT-2 | [`GPT2ForCausalLM`](@ref) | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| GPT-NeoX / Pythia | [`NeoXForCausalLM`](@ref) | `pythia-70m/410m/1b/6.9b` |
| Mixtral (sparse MoE) | [`MixtralForCausalLM`](@ref) | `Mixtral-8x7B/8x22B-Instruct-v0.1` |

Llama 3 RoPE scaling, Mistral and Gemma sliding-window attention, Gemma's logit
softcapping, Phi-3's fused QKV, NeoX's partial RoPE and parallel-residual block,
and Mixtral's top-k expert routing are all handled inside these types.

### Encoders and embeddings

| Family | Type | Parity recorded against |
| --- | --- | --- |
| BERT / RoBERTa | [`BertModel`](@ref), [`BertForMaskedLM`](@ref) | `bert-base-uncased`, `bert-base-cased`, `roberta-base`, `roberta-large` |
| Sentence embeddings | [`BertEmbeddingModel`](@ref) | `bge-small-en-v1.5`, `e5-small-v2` |
| Nomic | [`NomicBertModel`](@ref) | `nomic-embed-text-v1.5` |

### Vision, audio, and multimodal

| Family | Type | Parity recorded against |
| --- | --- | --- |
| ViT | [`ViTModel`](@ref), [`ViTForImageClassification`](@ref) | `vit-base-patch16-224` |
| SigLIP | [`SiglipModel`](@ref) | `siglip-base-patch16-224` |
| DINOv2 | [`Dinov2Model`](@ref) | `facebook/dinov2-base` |
| Whisper | [`WhisperModel`](@ref), [`transcribe`](@ref) | `openai/whisper-base` |
| LLaVA | [`LlavaForConditionalGeneration`](@ref) | unit tests only, no recorded fixture |

## Rough edges

- No `AutoModel` equivalent. You choose the config type and populate it.
- Generation is batch-1, and there is no continuous batching.
- Vision models take `pixel_values` directly. Image decoding and preprocessing
  are not in the package.
- Whisper's [`transcribe`](@ref) takes log-mel features, so the audio frontend
  is on you, and its decoder recomputes each step rather than using a KV cache,
  making cost quadratic in output length.
- LLaVA expects a vision tower that returns patch features with the CLS token
  already dropped. SigLIP fits; a CLIP tower is not implemented, which is what
  llava-1.5 end-to-end parity is waiting on.
- Mixtral runs each expert once over its assigned tokens without fusing the
  expert matmuls.

## License

MIT.
