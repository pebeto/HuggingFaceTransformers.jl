# Contributing to HuggingFaceTransformers.jl

Thanks for thinking about contributing. Read the guiding principles below
before opening a non-trivial PR, because they drive code review.

## Guiding principles

These exist because `Transformers.jl` accumulated friction over six years.
Re-read the list whenever a PR feels like it's growing.

1. **Lean dependency tree.** Every dependency must be either (a) widely used
   in the Julia ecosystem (Flux, NNlib, SafeTensors, JSON3, HTTP) or (b)
   vendored inline. No more chains of single-maintainer helper packages
   (`FuncPipelines`, `BangBang`, `Pickle`, `NeuralAttentionlib`,
   `BytePairEncoding`, `DoubleArrayTries`, `Fetch`, `ValSplit`,
   `WordTokenizers`, `TextEncodeBase`, `StructWalk`, `PrimitiveOneHot`,
   `Tricks`). Most of those are abandoned alongside Transformers.jl itself.
2. **Safetensors only.** No Pickle, no PyTorch `.bin`. If a model isn't
   published in safetensors, we don't load it. The whole ecosystem has moved.
3. **Generic loaders, not per-model branches.** A new model adds a config,
   layer wiring, and a state-dict mapping. It does not add a tokenizer
   subclass, a textencoder subclass, or a special download path. The hardcoded
   `bert_textencoder` / `gpt_textencoder` / `t5_textencoder` split in the old
   project blocked half the modern model lineup, and we don't repeat it.
4. **Decoder generation is a first-class feature, not an afterthought.**
   KV-cache, sampling, and chat templating belong on the core path rather than
   bolted on later.
5. **Numeric parity is the test.** Every model has a test that loads HF
   weights, runs a fixed input, and asserts logits match a recorded reference
   within tolerance. No parity test, no merge.
6. **No premature abstraction.** Three similar layers beat one configurable
   meta-layer. We can refactor once five exist.
7. **Modern Flux, Functors, and Zygote from day one.** Track current minors
   instead of carrying legacy version compatibility.

## Adding a model: the five-part contract

We don't merge a new model unless the PR ships all five. Splitting them
across PRs is fine; merging fewer than all five is not.

1. **Config struct.** A Julia struct mirroring the HF `config.json` keys
   you actually use. Anything not used isn't included.
2. **Layer wiring.** The forward pass, composed from
   `HuggingFaceTransformers.Layers`. Reuse existing layers if at all
   possible. If you need a new one, add it to `Layers` rather than burying
   it in the model file.
3. **State-dict map.** A pure-data table mapping HF parameter keys
   (e.g. `model.layers.0.self_attn.q_proj.weight`) to Julia layer
   addresses. The loader walks the table; no clever dispatch.
4. **Numeric-parity test.** Load the smallest public checkpoint from HF,
   run a fixed prompt, assert logits match a recorded reference output
   within tolerance. Generate the reference with `transformers` in Python
   and commit it under `test/fixtures/`. CI must fail when forward-pass
   output drifts.
5. **Runnable example.** A file in `examples/` that loads the model and
   does something interesting in under 30 lines.

If you find yourself adding a new tokenizer subclass, a new download or
cache path, or a new `*_textencoder` module, stop. This package does not
work that way; one of the existing generic paths needs to be extended
instead. Tag a maintainer.

## Style and formatting

- Code is formatted with [`JuliaFormatter`](https://domluna.github.io/JuliaFormatter.jl/)
  in BlueStyle. CI checks this; run `julia -e 'using JuliaFormatter; format(".", BlueStyle())'`
  locally before pushing.
- Public functions get a docstring. Internal helpers don't, unless the
  *why* of the function is non-obvious.
- Don't write comments that restate the code. Comment the surprise, the
  workaround, or the invariant, and nothing else.
- Docstrings are rendered by Documenter, so cross-reference other public
  symbols with `` [`name`](@ref) `` and keep code samples in fenced blocks.

## Documentation

The manual lives under `docs/src/` and is built with
[Documenter](https://documenter.juliadocs.org/). Build it locally with:

```
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build treats broken cross-references and undocumented exported symbols
as errors, so a PR that adds a public symbol needs to add it to the relevant
API page too. Output lands in `docs/build/`.

Pushes to `dev` publish to the `dev` docs, and pushes to `master` publish to
`stable`.

## Tests

- Every PR adds the relevant test(s); see the five-part contract above for
  model PRs.
- Tests run on CPU by default, with no network access and no model downloads.
  Run the suite with `julia --project -e 'using Pkg; Pkg.test()'`.
- Numeric-parity tests download real checkpoints, so each is opt-in through
  an environment variable: `HFT_TEST_PARITY` for Llama and
  `HFT_TEST_PARITY_<FAMILY>` for the rest. Each takes a variant label, a
  comma-separated list, or `all`.
- GPU tests are not part of the default suite. `test/gpu_attention.jl` runs
  by hand on a real device, and the plan for wiring GPU CI is in
  [`docs/src/gpu_ci.md`](docs/src/gpu_ci.md).

## Pull requests

- One logical change per PR.
- Title in the imperative ("Add Mistral attention", not "added Mistral
  attention").
- Update the docs alongside the code when a PR changes public behavior.

## Reporting bugs

GitHub Issues. Include the model name, a minimal reproducer, the Julia
version and OS, and the actual vs. expected output. If the bug is a
numeric mismatch with HF Python, include the Python script so we can
reproduce the reference.
