```@meta
CurrentModule = HuggingFaceTransformers
```

# Loading a checkpoint

One call covers the common case:

```julia
using HuggingFaceTransformers

lm = load("mistralai/Mistral-7B-Instruct-v0.3")
```

[`load`](@ref) downloads the repository, reads the architecture out of
`config.json`, builds the matching config and model, and loads the weights.
Keywords reach the model constructor, so an embedding checkpoint that wants mean
pooling is `load("intfloat/e5-small-v2"; pooling = :mean)`. Supported
architectures are listed in [`Models.AUTO_ARCHITECTURES`](@ref); anything else
throws and names what it found.

The rest of this page is the same work done by hand, which is what you want when
adding an architecture, changing a hyperparameter, or loading a checkpoint whose
`config.json` lies about its shape.

## 1. Fetch the repository

[`HFHub.snapshot_download`](@ref) resolves `org/repo[@revision]`, then fetches
`config.json`, the tokenizer files, and the weights into the same cache layout
the Python `huggingface_hub` library uses. Downloads are therefore shared
between your Python and Julia tooling, and cached blobs are reused.

```julia
using HuggingFaceTransformers.HFHub: snapshot_download

dir = snapshot_download("mistralai/Mistral-7B-Instruct-v0.3"; verbose=true)
```

Single files come through [`HFHub.download_file`](@ref) when you don't want the
whole snapshot. [`HFHub.default_cache_dir`](@ref) reports where things land, and
[`HFHub.parse_repo`](@ref) turns a string into a [`HFHub.RepoRef`](@ref).

## 2. Build the config

[`Models.config_from_json`](@ref) does this per architecture, and
[`Models.read_config`](@ref) parses the file:

```julia
using HuggingFaceTransformers.Models: MistralConfig, config_from_json, read_config

cfg = config_from_json(MistralConfig, read_config(dir))
```

Spelling it out is worth seeing once, because HF's keys and ours do not always
agree and some fields are derived rather than read:

```julia
using HuggingFaceTransformers.Models: MistralConfig
using JSON3

raw = JSON3.read(read(joinpath(dir, "config.json"), String))

cfg = MistralConfig(;
    vocab_size=Int(raw.vocab_size),
    hidden_size=Int(raw.hidden_size),
    intermediate_size=Int(raw.intermediate_size),
    num_hidden_layers=Int(raw.num_hidden_layers),
    num_attention_heads=Int(raw.num_attention_heads),
    num_key_value_heads=Int(raw.num_key_value_heads),
    head_dim=Int(raw.hidden_size) ÷ Int(raw.num_attention_heads),
    max_position_embeddings=Int(raw.max_position_embeddings),
    rope_theta=Float64(raw.rope_theta),
    rms_norm_eps=Float64(raw.rms_norm_eps),
)
```

The field names mirror the HF keys, Julianized to snake_case, and a config only
carries the keys the implementation actually reads.

## 3. Construct the model

Calling the model type on its config allocates the parameters.

```julia
using HuggingFaceTransformers.Models: MistralForCausalLM

lm = MistralForCausalLM(cfg)
```

## 4. Load the weights

[`Models.load_weights`](@ref) reads a safetensors checkpoint into a dictionary
keyed on HF parameter names, and [`Models.load_state_dict!`](@ref) walks a
pure-data mapping table to place each tensor.

```julia
using HuggingFaceTransformers.Models: load_weights, load_state_dict!

load_state_dict!(lm, load_weights(dir))
```

`load_weights` accepts a directory, a single `.safetensors` file, or a
`model.safetensors.index.json` for sharded checkpoints. Tensors come back as
lazy views over a memory map by default, so nothing is read until a parameter
is filled from it; pass `mmap=false` to read everything up front, which helps
when the cache is on slow storage and the weights are headed to a GPU anyway.

`snapshot_download` honours `HF_HUB_OFFLINE=1` as the Python library does,
returning the cached snapshot without touching the network. It never modifies
an entry already in a snapshot, so a cache written by another user or tool is
safe to load from.
PyTorch `.bin` pickles are not read at all.

## GGUF files

llama.cpp's GGUF files load through the same front door. Name the file in the
repository, and only that file downloads:

```julia
lm = load("Qwen/Qwen2.5-0.5B-Instruct-GGUF"; gguf_file="qwen2.5-0.5b-instruct-q4_k_m.gguf")
lm = load("path/to/model.gguf")
```

[`Models.load_gguf_model`](@ref) reads the architecture and hyperparameters from
the file's metadata and maps llama.cpp's tensor names (`blk.0.attn_q.weight` and
so on) onto the HF names the state-dict tables expect, so a GGUF model is the
same model type a safetensors checkpoint builds. It covers the `llama` (Llama 2
and 3.x, Mistral, SmolLM, TinyLlama), `qwen2`, `qwen3` and `gemma2`
architectures. Two conversions llama.cpp makes are undone on the way in: the
llama family's query and key rows are reordered for its rotary layout, and Gemma
2's norm weights carry a folded-in 1.

Every tensor type in the common quantizations decodes: F32, F16, BF16, the
legacy Q4_0 to Q8_0 blocks, the k-quants Q2_K to Q6_K, and IQ4_NL, which
llama.cpp substitutes in rows too short for a k-quant block.

The weights stay in the file's encoding. Each projection matrix becomes a
[`Models.GGUFMatrix`](@ref) over the memory-mapped file, decoded block by block
as it is multiplied, and the embedding a [`Models.GGUFEmbedding`](@ref) that
decodes only the rows of the tokens it looks up. Only the norms and biases are
decoded up front, so a model takes about its file size in memory: Qwen2.5-7B at
Q4_K_M peaks at 5.2 GB and generates at about 0.13 s per token on a 12-core CPU.
This path runs on the CPU. `load(...; packed=false)` dequantizes every tensor to
`Float32` instead, for a dense model that trains and moves to a GPU like any
other, at the full-precision model's memory. Both ways, the logits agree with
`transformers` loading the same file to within 2e-5.

The file also carries its tokenizer, chat template and stop tokens, so one
download is enough. [`Tokenizers.load_tokenizer`](@ref),
[`Generation.load_chat_template`](@ref) and
[`Generation.load_generation_config`](@ref) all take the path of a `.gguf` file:

```julia
path = HFHub.download_file("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
lm, tokenizer = load(path), load_tokenizer(path)
prompt = load_chat_template(path)([Dict("role" => "user", "content" => "Hi")]; add_generation_prompt=true)
generate(lm, encode(tokenizer, prompt); generation_config=load_generation_config(path))
```

The tokenizer is rebuilt from llama.cpp's metadata. Byte-level BPE files
(`tokenizer.ggml.model` is `gpt2`) use the split rules of their pre-tokenizer,
which covers Llama 3 (`llama-bpe`), Qwen2 and Qwen3 (`qwen2`), SmolLM and plain
GPT-2; other pre-tokenizers raise. SentencePiece files (`llama`, as Llama 2,
Mistral, TinyLlama and Gemma 2 store theirs) keep scores rather than merges, and
the merge order is derived from them. Like llama.cpp, a space is prefixed to
every run of text, also right after a control token. On a set of strings with
accents, CJK, emoji, digits, code and whitespace runs, the GGUF tokenizers of
SmolLM2, Qwen2.5, Qwen3, Gemma 2, TinyLlama and Llama 3.2 encode and decode
exactly as the original repositories' `tokenizer.json` does. Some older
conversions wrote every SentencePiece score as 0, which loses the merge order;
TheBloke's TinyLlama is one, and its whitespace runs split differently.

The stop tokens are the ones llama.cpp ends generation on: the EOS, any
end-of-turn token the file names, and turn-end control tokens such as
`<end_of_turn>` and `<|im_end|>`. For Gemma 2, Qwen2.5 and Llama 3.2 that is the
list their `generation_config.json` gives. GGUF holds no sampling settings.

Llama 3.1 and later scale RoPE, and GGUF stores the result as one divisor per
rotary frequency (`rope_freqs.weight`) rather than the scaling parameters; each
frequency is divided by its factor, as llama.cpp does. `transformers` skips these
factors when it loads a GGUF, so this path is checked against the original
safetensors instead: Llama-3.2-1B-Instruct's F16 GGUF matches its safetensors
checkpoint to within 2e-5 on a 702-token prompt, where leaving the factors out is
off by 1.4. Other RoPE scaling types in GGUF metadata raise. Llama 3's tokenizer
takes a pre-token whole when it is in the vocabulary (`ignore_merges`), from GGUF
and from `tokenizer.json` alike.

## Inspecting the wiring

Every family exposes its mapping table, so you can see exactly which HF
parameter lands where and how it is transformed.

```julia
using HuggingFaceTransformers.Models: mistral_state_dict_map

map = mistral_state_dict_map(cfg)
map["model.layers.0.self_attn.q_proj.weight"]
```

Each entry is a `(path, transform)` pair: `path` addresses a field inside the
model and `transform` is `:as_is`, `:transpose`, or another symbol handled by
the loader. [`Models.load_into!`](@ref) applies a table to any target, which is
what the per-model `load_state_dict!` methods call.
