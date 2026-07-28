# HuggingFaceTransformers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pebeto.github.io/HuggingFaceTransformers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pebeto.github.io/HuggingFaceTransformers.jl/dev)

Load and run HuggingFace transformer checkpoints in Julia. Weights come from
the Hub in safetensors format, tokenizers from `tokenizer.json`, and the
forward passes are plain Flux layers, so they differentiate and move to a GPU
like any other Flux model.

The spiritual successor to
[`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl), rebuilt
with a leaner dependency tree and a focus on decoder LLMs.

Every architecture below except LLaVA has a numeric-parity test that compares
logits against a reference recorded from `transformers` in Python. The CPU
suite is 1463 assertions and runs in about 90 seconds.

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

## Loading a checkpoint

Loading is four explicit steps, and the middle one is manual:

1. `snapshot_download(repo_id)` resolves the revision and fetches
   `config.json`, the tokenizer files, and the weights into the Python-compatible
   cache layout. Already-cached blobs are reused.
2. You build the config struct yourself from `config.json`. There is no
   `AutoConfig` equivalent that guesses the architecture, so you pick the
   `*Config` type and fill in the fields. Each file under `examples/` shows
   this for one family.
3. Construct the model from the config, which allocates the parameters.
4. `load_state_dict!(model, load_weights(dir))` walks a pure-data mapping table
   from HF parameter names to Julia layer addresses, transposing or slicing
   where the layouts differ.

`load_weights` accepts a directory, a single `.safetensors` file, or a
`model.safetensors.index.json` for sharded checkpoints, and returns
mmap-backed views by default. Pass `mmap=false` to copy eagerly. PyTorch
`.bin` pickles are not read at all.

Every family also exposes its mapping table (`llama_state_dict_map(cfg)` and
friends) if you want to inspect or extend the wiring.

## Supported architectures

### Decoder language models

| Family | Type | Parity recorded against |
| --- | --- | --- |
| Llama 3.x | `LlamaForCausalLM` | `Llama-3.2-1B/3B-Instruct`, `Llama-3.1-8B/70B-Instruct` |
| Mistral | `MistralForCausalLM` | `Mistral-7B-Instruct-v0.2`, `v0.3` |
| Qwen2 / Qwen2.5 | `QwenForCausalLM` | `Qwen2.5-0.5B/1.5B/3B/7B-Instruct` |
| Gemma 2 | `GemmaForCausalLM` | `gemma-2-2b/9b/27b-it` |
| Phi-3 | `Phi3ForCausalLM` | `Phi-3-mini-4k-instruct`, `Phi-3-medium-4k-instruct` |
| GPT-2 | `GPT2ForCausalLM` | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| GPT-NeoX / Pythia | `NeoXForCausalLM` | `pythia-70m/410m/1b/6.9b` |
| Mixtral (sparse MoE) | `MixtralForCausalLM` | `Mixtral-8x7B/8x22B-Instruct-v0.1` |

Llama 3 RoPE scaling, Mistral and Gemma sliding-window attention, Gemma's
logit softcapping, Phi-3's fused QKV, NeoX's partial RoPE and parallel-residual
block, and Mixtral's top-k expert routing are all handled inside these types.

### Encoders and embeddings

| Family | Type | Parity recorded against |
| --- | --- | --- |
| BERT / RoBERTa | `BertModel`, `BertForMaskedLM` | `bert-base-uncased`, `bert-base-cased`, `roberta-base`, `roberta-large` |
| Sentence embeddings | `BertEmbeddingModel` | `bge-small-en-v1.5`, `e5-small-v2` |
| Nomic | `NomicBertModel` | `nomic-embed-text-v1.5` |

### Vision, audio, and multimodal

| Family | Type | Parity recorded against |
| --- | --- | --- |
| ViT | `ViTModel`, `ViTForImageClassification` | `vit-base-patch16-224` |
| SigLIP | `SiglipModel` (vision and text towers) | `siglip-base-patch16-224` |
| DINOv2 | `Dinov2Model` | `facebook/dinov2-base` |
| Whisper | `WhisperModel`, `transcribe` | `openai/whisper-base` |
| LLaVA | `LlavaForConditionalGeneration` | unit tests only, no recorded fixture |

## Generation

`generate` mirrors the keyword arguments of HuggingFace's `model.generate`:

```julia
ids = generate(
    lm,
    prompt_ids;
    max_new_tokens=256,
    do_sample=true,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=[128001, 128009],
)
```

Token IDs are 0-indexed to match HF, and the return value is the full
`[prompt..., generated...]` sequence. The two-argument form
`generate(lm, tokenizer, prompt)` tokenizes and detokenizes around it.
Generation is batch-1.

The KV-cache is preallocated as
`(head_dim, n_kv_heads, max_seq, batch)` and mutated in place, feature-first so
that cache writes are zero-copy slice assignments. `generate` manages it for
you; drive it yourself when you need the logits:

```julia
caches = build_caches(lm, max_seq, 1)
logits = lm(reshape(prompt_ids, :, 1); caches=caches, step=1)
reset!(caches)
```

### Chat templates

`ChatTemplate` renders the `chat_template` field out of `tokenizer_config.json`
through a small Jinja2 interpreter, which covers the Llama-3, Qwen2.5, and
Mistral templates:

```julia
tk_cfg = JSON3.read(read(joinpath(dir, "tokenizer_config.json"), String))
template = ChatTemplate(String(tk_cfg.chat_template))

prompt = template(
    [
        Dict("role" => "system", "content" => "You are a terse assistant."),
        Dict("role" => "user", "content" => "What is the capital of Peru?"),
    ];
    add_generation_prompt=true,
    bos_token="<|begin_of_text|>",
)
```

Supported: `if`/`elif`/`else`, `for`, `set`, attribute and item access,
slicing, arithmetic and comparison and logical operators, `in`, `is defined`,
the `trim` and `tojson` filters, `raise_exception`, the `loop.*` variables, and
whitespace control. Not supported: tuple, list, and dict literals, float
literals, macros, includes, custom filters, custom tests, and step slicing.
Templates that reach for those raise on construction, which is why the
examples carry a plain fallback template.

### Speculative decoding

```julia
ids = speculative_generate(target, draft, prompt_ids; n_draft=4, max_new_tokens=64)
```

The draft model proposes `n_draft` tokens, the target verifies them in one
forward pass, and the accept/reject rule keeps the output distributed exactly
as the target alone would produce. Greedy speculative decoding reproduces
`generate(target, ...)` byte for byte. The two models must share a vocabulary.
`top_k`, `top_p`, and `repetition_penalty` are not applied on this path yet.

## Embeddings

```julia
model = BertEmbeddingModel(cfg; pooling=:mean, normalize=true)
load_state_dict!(model, load_weights(dir))

v = embed(model, tokenizer, "hello world")
```

`pooling` takes `:cls` or `:mean`, and `normalize=true` applies L2. The
`mean_pool`, `cls_pool`, and `l2_normalize` functions are exported if you want
to pool the trunk output yourself.

## Tokenizers

`load_tokenizer(path)` reads `tokenizer.json` and handles three model types
with no per-model subclasses:

- BPE with ByteLevel pre-tokenization, for the GPT-2, Llama-3, Qwen2, and
  RoBERTa family.
- Unigram with Metaspace pre-tokenization, for SentencePiece checkpoints such
  as Gemma.
- WordPiece with the BERT normalizer, for BERT and its descendants.

Older checkpoints that ship only a `vocab.txt` load through
`load_wordpiece_from_vocab_txt(path)`. Round-trip behavior is checked against
Python `AutoTokenizer` output on a shared prompt set covering accents, CJK,
Korean, emoji, and whitespace edge cases.

## Precision and quantization

```julia
lm16 = fp16(lm)          # also bf16(lm), fp32(lm), convert_eltype(lm, T)
lmq  = quantize_int8(lm) # per-row symmetric int8, weight-only
```

The conversions return a new model and leave the original alone. Norms and
token embeddings stay Float32 under `quantize_int8`, weights shrink about 4x,
and the matmul dequantizes per call. Expect logit error around `1e-2` to `1e-1`
against the fp32 reference, growing with depth. The docstring on
`convert_eltype` records the per-dtype tolerance bands.

## GPU

Move a model with Flux and the rest follows:

```julia
using Flux: gpu
using CUDA   # or AMDGPU, or Metal on Apple Silicon

lm_gpu = gpu(lm)
```

Importing a backend loads the matching package extension, which routes `sdpa`
on device arrays to `flash_sdpa`, a tiled attention kernel that never
materializes the full score matrix. The CPU suite checks the tiled path against
the materialized one on causal, sliding-window, softcap, and multi-block
inputs. `test/gpu_attention.jl` runs the same comparison on a real device and
is not part of the default suite:

```
HFT_GPU_BACKEND=cuda julia --project=. test/gpu_attention.jl
```

## Fine-tuning

LoRA adapters, gradient checkpointing, and PEFT-format saving are here; a
training framework is not. `examples/lora_finetune.jl` is a plain Flux loop
that ties them together.

```julia
lora = lora_wrap(base_linear, 8, 16.0)   # rank 8, alpha 16
merged = merge_lora(lora)                # fold A*B back into one Linear

save_lora(
    adapter_dir,
    LoraConfig(; r=8, lora_alpha=16.0, target_modules=["q_proj", "v_proj"]),
    Dict("model.layers.0.self_attn.q_proj" => (lora.lora_A, lora.lora_B)),
)

adapter_cfg, adapter_weights = load_lora(adapter_dir)
n_applied = apply_lora!(model, adapter_dir, llama_state_dict_map(cfg))
```

The LoRA functions live in `HuggingFaceTransformers.Models`, while `Linear` and
`Checkpointed` come from `HuggingFaceTransformers.Layers`.

`Checkpointed(layer)` recomputes activations in the backward pass instead of
storing them. It is transparent to Flux and Functors, and results and gradients
match the uncheckpointed path exactly, so it buys memory rather than accuracy.

## GGUF

`load_gguf(path)` reads a llama.cpp GGUF file and returns its metadata plus
tensors dequantized to `Float32`, keeping GGML's contiguous-first dimension
order, which lines up with Julia's column-major layout. F32, F16, Q8_0, and
Q4_0 are dequantized. The k-quants (Q4_K, Q5_K, Q6_K and friends) throw an
error naming the dtype rather than returning garbage. Mapping a GGUF file onto
one of the model types above is not wired up, so this is a loader, not yet a
second checkpoint format.

## Examples

Each file runs standalone with `julia --project=. examples/<file>`, and every
one except `lora_finetune.jl` takes a repository ID as an optional first
argument.

| File | What it does |
| --- | --- |
| `repl_chat.jl` | Llama-3.x chat, including the download and chat templating |
| `repl_chat_mistral.jl` | Mistral chat with the `[INST]` template |
| `repl_chat_qwen.jl` | Qwen2 / Qwen2.5 chat |
| `repl_chat_gemma.jl` | Gemma 2 chat |
| `repl_chat_phi3.jl` | Phi-3 chat |
| `repl_chat_mixtral.jl` | Mixtral chat |
| `completion_gpt2.jl` | GPT-2 prefix completion, no chat roles |
| `completion_neox.jl` | Pythia prefix completion |
| `mask_fill_roberta.jl` | RoBERTa masked-token filling |
| `lora_finetune.jl` | LoRA training loop with gradient checkpointing, on a small random model so it needs no download |

## Tests

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

That runs the CPU suite, which needs no network and no model downloads.
Numeric-parity tests are separate: they download real checkpoints and compare
logits against a fixture recorded from Python, so each is opt-in through an
environment variable.

```
HFT_TEST_PARITY=1B          julia --project -e 'using Pkg; Pkg.test()'
HFT_TEST_PARITY_QWEN=all    julia --project -e 'using Pkg; Pkg.test()'
```

The gates are `HFT_TEST_PARITY` for Llama and `HFT_TEST_PARITY_<FAMILY>` for
the rest, each taking a variant label, a comma-separated list, or `all`.
Unknown labels fail loudly with the valid set, and missing fixtures skip with
the exact recorder command to run. Recorders live in `test/fixtures/` as small
Python scripts. `HFT_TEST_JET=1` adds a JET smoke pass, and
`HFT_TEST_NETWORK=1` enables the Hub integration test.

## Rough edges

- No `AutoModel` equivalent. You choose the config type and populate it.
- Generation is batch-1, and there is no continuous batching.
- Vision models take `pixel_values` directly. Image decoding and preprocessing
  are not in the package.
- Whisper's `transcribe` takes log-mel features, so the audio frontend is on
  you, and its decoder recomputes each step rather than using a KV cache,
  making cost quadratic in output length.
- LLaVA expects a vision tower that returns patch features with the CLS token
  already dropped. SigLIP fits; a CLIP tower is not implemented, which is what
  llava-1.5 end-to-end parity is waiting on.
- Mixtral runs each expert once over its assigned tokens without fusing the
  expert matmuls.

## Contributing

A new model is a five-part contract: config struct, layer wiring, state-dict
map, numeric-parity test, and a runnable example. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT.
