# HuggingFaceTransformers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pebeto.github.io/HuggingFaceTransformers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pebeto.github.io/HuggingFaceTransformers.jl/dev)

Load and run Hugging Face transformer checkpoints in Julia. Weights come from
safetensors or llama.cpp GGUF files, tokenizers from `tokenizer.json`, and the
models are plain Flux layers, so they differentiate and move to a GPU like any
other Flux model.

It is a successor to
[Transformers.jl](https://github.com/chengchingwen/Transformers.jl), with fewer
dependencies and a focus on decoder LLMs. Each supported architecture is tested
for numeric parity: its logits are compared against a reference recorded from
Python `transformers`.

This is a personal project that I work on in my spare time. It is not
registered yet, and the API may still change.

## Installation

Requires Julia 1.10 or newer.

```julia
pkg> dev https://github.com/pebeto/HuggingFaceTransformers.jl
```

## Quick start

```julia
using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer
using HuggingFaceTransformers.Generation: generate

lm = load("gpt2")
tokenizer = load_tokenizer(snapshot_download("gpt2"))

println(generate(lm, tokenizer, "The Julia language is"; max_new_tokens=32))
```

`load` reads `config.json` to choose the architecture. GGUF files load the same
way, and stay quantized in memory:

```julia
lm = load("Qwen/Qwen2.5-0.5B-Instruct-GGUF"; gguf_file="qwen2.5-0.5b-instruct-q4_k_m.gguf")
```

Gated repositories such as Llama and Gemma read the token from `HF_TOKEN` or
from the file that `huggingface-cli login` writes.

## Supported models

| Kind | Families |
| --- | --- |
| Decoders | Llama 3.x, Mistral, Mixtral, Qwen2/2.5/3, Qwen3.5 (text), Gemma 2, Phi-3, GPT-2, GPT-NeoX/Pythia, gpt-oss |
| Encoders | BERT, RoBERTa, BGE/E5 sentence embeddings, Nomic Embed |
| Vision and audio | ViT, SigLIP, DINOv2, Whisper, LLaVA |
| GGUF | `llama` (Llama 2 and 3.x), `qwen2`, `qwen3`, `gemma2`, from F32 down to Q2_K |

The [documentation](https://pebeto.github.io/HuggingFaceTransformers.jl/dev)
lists the checkpoints each family was checked against and the known
limitations.

## Features

- [Generation](https://pebeto.github.io/HuggingFaceTransformers.jl/dev/guide/generation/):
  sampling, batching, streaming, stop strings, chat templates, speculative
  decoding and Whisper transcription.
- [Quantization](https://pebeto.github.io/HuggingFaceTransformers.jl/dev/guide/performance/):
  fp16/bf16, int8, and packed NVFP4, FP8, MXFP4 and GGUF weights decoded as they
  are multiplied.
- [GPU](https://pebeto.github.io/HuggingFaceTransformers.jl/dev/guide/performance/#GPU):
  through CUDA, AMDGPU or Metal, with a tiled attention kernel.
- [Fine-tuning](https://pebeto.github.io/HuggingFaceTransformers.jl/dev/guide/finetuning/):
  LoRA adapters in PEFT format and gradient checkpointing.
- [Tokenizers](https://pebeto.github.io/HuggingFaceTransformers.jl/dev/guide/tokenizers/)
  and [image processors](https://pebeto.github.io/HuggingFaceTransformers.jl/dev/guide/images/)
  that match their Python counterparts output for output.

Runnable scripts are in [`examples/`](examples).

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```

The default suite runs on the CPU, with no network access. The parity tests
download real checkpoints, so they are opt-in; see
[`CONTRIBUTING.md`](CONTRIBUTING.md#tests).

## Contributing

Each new model comes with a config, its layers, a state-dict map, a parity test
and an example. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the details.

## License

MIT.
