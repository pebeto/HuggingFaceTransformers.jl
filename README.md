# Allspark.jl

A maintained Julia library for loading and running HuggingFace transformer
models. Spiritual successor to
[`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl).

> **Status: pre-alpha.** The API shown below is *aspirational*. The project
> bootstrap is done; nothing else works yet. See
> [`ROADMAP.md`](ROADMAP.md) for the build plan.

## Target API

What the library is *aiming* to look like once the core path lands:

```julia
using Allspark

tokenizer, model = Allspark.load("meta-llama/Llama-3.2-1B-Instruct")

reply = Allspark.generate(
    model, tokenizer,
    "Explain why grass is green in one sentence.";
    max_new_tokens = 64,
    temperature = 0.7,
    top_p = 0.9,
)

println(reply)
```

Chat-style API:

```julia
messages = [
    (role = "system", content = "You are a terse assistant."),
    (role = "user",   content = "What is the capital of Peru?"),
]

reply = Allspark.chat(model, tokenizer, messages; max_new_tokens = 32)
```

For encoder workflows (embeddings, classification):

```julia
tokenizer, model = Allspark.load("BAAI/bge-small-en-v1.5")

embeddings = Allspark.embed(model, tokenizer, ["hello world", "goodbye world"])
```

## Installation

Not registered yet. To follow along:

```julia
julia> ]
pkg> dev https://github.com/<owner>/Allspark.jl
```

## Contributing

Adding a new model is a five-part contract: config struct, layer wiring,
state-dict map, numeric-parity test, runnable example. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the details.

## License

MIT.
