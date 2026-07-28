```@meta
CurrentModule = HuggingFaceTransformers
```

# Loading a checkpoint

Loading is four explicit steps, and the middle one is manual.

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

There is no `AutoConfig` equivalent that guesses the architecture from
`config.json`, so you pick the config type and fill in the fields you need. Each
file under `examples/` does this for one family.

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
mmap-backed views by default; pass `mmap=false` to copy eagerly, which helps
when the cache is on slow storage and the weights are headed to a GPU anyway.
PyTorch `.bin` pickles are not read at all.

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
