```@meta
CurrentModule = HuggingFaceTransformers
```

# Fine-tuning

LoRA adapters, gradient checkpointing, and PEFT-format saving are here. A
training framework is not: `examples/lora_finetune.jl` is a plain Flux loop that
ties the pieces together, and it runs on a small random model so it needs no
download.

## LoRA adapters

[`Models.lora_wrap`](@ref) wraps a [`Layers.Linear`](@ref) in a
[`Models.LoRALinear`](@ref), freezing the base weight and training only the
low-rank `A` and `B` factors.

```julia
using HuggingFaceTransformers.Models: lora_wrap, merge_lora

lora = lora_wrap(base_linear, 8, 16.0)   # rank 8, alpha 16
merged = merge_lora(lora)                # fold scaling * B*A back into one Linear
```

[`Models.lora_scaling`](@ref) computes the multiplier on the update:
`lora_alpha / r` normally, or `lora_alpha / sqrt(r)` when
`use_rslora=true` keeps the update's magnitude stable as rank grows.

Swap every projection in a model with `Functors.fmap`. The `exclude` predicate
matters: without it `fmap` descends into the `Linear` itself and never offers you
the layer to replace.

```julia
using Flux, Functors
using Functors: isleaf

lora_wrap_all(model) =
    fmap(model; exclude=(x -> x isa Linear || isleaf(x))) do x
        x isa Linear ? lora_wrap(x, 8, 16.0) : x
    end

model = lora_wrap_all(base)
```

Freezing happens on the optimiser state rather than the model, so build the state
first and freeze through it:

```julia
opt = Flux.setup(Flux.Adam(0.01), model)

Flux.freeze!(opt.model.embed_tokens)
Flux.freeze!(opt.model.norm)
for layer in opt.model.layers
    Flux.freeze!(layer.input_layernorm)
    Flux.freeze!(layer.post_attention_layernorm)
end
```

With the base weights frozen and `LoRALinear` declaring only `lora_A` and
`lora_B` trainable, `withgradient` and `update!` then train the adapters alone.
`examples/lora_finetune.jl` runs exactly this loop end to end.

## Saving and loading adapters

[`Models.save_lora`](@ref) writes a PEFT-format adapter directory
(`adapter_config.json` plus `adapter_model.safetensors`), so the result loads in
the Python ecosystem too. [`Models.load_lora`](@ref) reads one back, and
[`Models.apply_lora!`](@ref) merges an adapter into a base model in place,
returning how many modules it touched.

```julia
using HuggingFaceTransformers.Models: LoraConfig, save_lora, load_lora, apply_lora!

save_lora(
    adapter_dir,
    LoraConfig(; r=8, lora_alpha=16.0, target_modules=["q_proj", "v_proj"]),
    Dict("model.layers.0.self_attn.q_proj" => (lora.lora_A, lora.lora_B)),
)

adapter_cfg, adapter_weights = load_lora(adapter_dir)
n_applied = apply_lora!(model, adapter_dir, llama_state_dict_map(cfg))
```

`apply_lora!` locates each target's base weight through the state-dict map, so
pass the map for the model you are patching. It refuses to patch a weight the
map marks as transposed, rather than silently applying the update to the wrong
orientation.

## Gradient checkpointing

[`Layers.Checkpointed`](@ref) wraps a layer so its forward is recomputed during
the backward pass instead of storing activations. It is built on
`ChainRulesCore`, so it stays AD-agnostic, and it is transparent to Flux and
Functors: the wrapper's parameters are the wrapped layer's.

```julia
using HuggingFaceTransformers.Layers: Checkpointed, checkpoint

layer = Checkpointed(GeluMLP(hidden, intermediate))
```

Results and gradients match the uncheckpointed path exactly, so this buys memory
for deep stacks rather than changing numerics. [`Layers.checkpoint`](@ref) is the
function form when you want to checkpoint an arbitrary call rather than a layer.
