#!/usr/bin/env julia
# HuggingFaceTransformers.jl LoRA fine-tuning example — a plain Flux training loop, not a
# framework. It ties together three pieces: LoRA adapters (train only low-rank
# A/B, freeze the base), gradient checkpointing (recompute activations in the
# backward pass to save memory), and PEFT-format adapter saving.
#
# Self-contained: it builds a small randomly-initialized Llama and overfits a
# fixed token sequence, so it runs in seconds with no downloads. Swap the model
# construction for `load_state_dict!(lm, load_weights(snapshot_download(repo)))`
# to fine-tune a real checkpoint.
#
# Usage:
#   julia --project=. examples/lora_finetune.jl

using Random
using Flux
using Functors: isleaf
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Layers: Linear, checkpoint

const VOCAB = 32
const SEQLEN = 16
const RANK = 8
const ALPHA = 16.0

# Replace every `Linear` in the model with a LoRA-wrapped version. The base
# weight is frozen inside `LoRALinear` (its `trainable` is only `lora_A`/`lora_B`),
# so this is parameter-efficient by construction.
lora_wrap_all(model) =
    fmap(model; exclude=(x -> x isa Linear || isleaf(x))) do x
        x isa Linear ? lora_wrap(x, RANK, ALPHA) : x
    end

# Next-token cross-entropy over a (seq, batch) batch of token IDs. The forward
# is wrapped in `checkpoint` so the whole network's activations are recomputed in
# the backward pass rather than stored (coarse checkpointing; wrap individual
# layers for a finer memory/compute balance).
function nexttoken_loss(model, ids)
    logits = checkpoint(model, ids)                 # (vocab, seq, batch)
    pred = reshape(logits[:, 1:(end - 1), :], VOCAB, :)
    targets = Flux.onehotbatch(vec(ids[2:end, :]), 0:(VOCAB - 1))
    return Flux.logitcrossentropy(pred, targets)
end

function main()
    Random.seed!(1234)

    cfg = LlamaConfig(;
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=SEQLEN,
        rope_theta=10000.0,
        tie_word_embeddings=false,
    )
    base = LlamaForCausalLM(cfg)
    model = lora_wrap_all(base)

    # A single fixed sequence to memorize (1 example, batch of 1).
    data = reshape(rand(0:(VOCAB - 1), SEQLEN), :, 1)

    opt = Flux.setup(Flux.Adam(0.01), model)
    # Freeze everything that isn't a LoRA parameter: token embeddings and the
    # RMSNorms. (The base of each LoRALinear is already frozen.)
    Flux.freeze!(opt.model.embed_tokens)
    Flux.freeze!(opt.model.norm)
    for layer in opt.model.layers
        Flux.freeze!(layer.input_layernorm)
        Flux.freeze!(layer.post_attention_layernorm)
    end

    println(
        "Fine-tuning a $(cfg.num_hidden_layers)-layer LoRA (rank $(RANK)) to memorize a sequence...",
    )
    init_loss = nexttoken_loss(model, data)
    for epoch in 1:100
        loss, grads = Flux.withgradient(m -> nexttoken_loss(m, data), model)
        Flux.update!(opt, model, grads[1])
        (epoch == 1 || epoch % 20 == 0) &&
            println("  epoch $(lpad(epoch, 3))   loss $(round(loss; digits=4))")
    end
    final_loss = nexttoken_loss(model, data)
    println("loss: $(round(init_loss; digits=4)) → $(round(final_loss; digits=4))")

    # Collect the trained LoRA matrices per attention module and save a
    # PEFT-format adapter.
    module_weights = Dict{String,Tuple{Matrix{Float32},Matrix{Float32}}}()
    for (i, layer) in enumerate(model.model.layers)
        for (field, name) in
            ((:wq, "q_proj"), (:wk, "k_proj"), (:wv, "v_proj"), (:wo, "o_proj"))
            ll = getfield(layer.self_attn, field)
            module_weights["model.layers.$(i - 1).self_attn.$(name)"] = (
                ll.lora_A, ll.lora_B
            )
        end
    end
    out_dir = mktempdir()
    save_lora(
        out_dir,
        LoraConfig(;
            r=RANK,
            lora_alpha=ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        module_weights,
    )
    println("Saved PEFT adapter to $(out_dir)")
    return println("  files: ", join(readdir(out_dir), ", "))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
