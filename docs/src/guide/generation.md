```@meta
CurrentModule = HuggingFaceTransformers
```

# Generation

[`Generation.generate`](@ref) mirrors the keyword arguments of HuggingFace's
`model.generate`. Token IDs are 0-indexed to match HF, and the return value is
the full `[prompt..., generated...]` sequence. Generation is batch-1.

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

Greedy decoding is the default. Passing `do_sample=false` ignores
`temperature`, `top_k`, `top_p`, and `rng` entirely. The repetition penalty
follows the HF rule: logits at already-seen IDs are divided when the penalty is
positive and multiplied when it is negative.

There is a two-argument form that tokenizes and detokenizes around the call,
which is the one to reach for in a REPL:

```julia
reply = generate(lm, tokenizer, "Explain why grass is green."; max_new_tokens=64)
```

## Driving the KV-cache yourself

[`Layers.KVCache`](@ref) is preallocated as
`(head_dim, n_kv_heads, max_seq, batch)` and mutated in place. The feature-first
layout means cache writes are zero-copy slice assignments rather than
`permutedims` round trips.

`generate` allocates and manages the cache for you. Call the model directly when
you want the logits:

```julia
caches = build_caches(lm, max_seq, 1)

logits = lm(reshape(prompt_ids, :, 1); caches=caches, step=1)   # prefill
next = lm(reshape([token], 1, 1); caches=caches, step=length(prompt_ids) + 1)

reset!(caches)   # zero the cache to reuse it for another sequence
```

[`Models.build_caches`](@ref) has a method per architecture, so the cache is
sized from that model's head count and head dimension.

## Chat templates

[`Generation.ChatTemplate`](@ref) renders the `chat_template` field out of
`tokenizer_config.json` through a small Jinja2 interpreter, which covers the
Llama-3, Qwen2.5, and Mistral templates.

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

Supported: `if`/`elif`/`else`, `for`, `set`, attribute and item access, slicing,
arithmetic and comparison and logical operators, `in`, `is defined`, the `trim`
and `tojson` filters, `raise_exception`, the `loop.*` variables, and whitespace
control.

Not supported: tuple, list, and dict literals, float literals, macros,
includes, custom filters, custom tests, and step slicing. A template that
reaches for those raises on construction, which is why the examples carry a
plain fallback template and warn when they fall back to it.

[`Generation.apply_chat_template`](@ref) is the one-shot form when you don't
need to keep the parsed template around.

## Speculative decoding

```julia
ids = speculative_generate(target, draft, prompt_ids; n_draft=4, max_new_tokens=64)
```

The small `draft` model proposes `n_draft` tokens autoregressively, the large
`target` verifies them in a single forward pass, and the accept/reject rule
keeps the output distributed exactly as `target` alone would produce it. Greedy
speculative decoding reproduces `generate(target, ...)` byte for byte, which is
what the test suite asserts.

The two models must share a vocabulary, since token IDs are compared directly;
a mismatched LM-head width raises an error. `top_k`, `top_p`, and
`repetition_penalty` are not applied on this path yet.

Speedup tracks the draft's acceptance rate, because the win is verifying
`n_draft` tokens per target forward instead of one. Each round also sync-feeds
the last committed token to both models before drafting, and that single-token
pass is overhead a tuned implementation would fold into the next verify pass.

## Multimodal generation

[`Models.generate_multimodal`](@ref) runs the LLaVA path: image features are
projected and spliced into the text embeddings at the image-token positions,
then the language model generates over the merged sequence. Use
[`Models.forward_from_features`](@ref) when you have already computed the vision
features and only want logits.
