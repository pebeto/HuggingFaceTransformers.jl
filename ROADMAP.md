# Allspark.jl Roadmap

A modern, maintained Julia package for loading and running HuggingFace
transformer models — the spiritual successor to `Transformers.jl`, rebuilt with
fewer dependencies, a tighter scope, and a "decoder LLMs first" attitude.

This document defines the build order. Phase 0 and Phase 1 are sequential and
on the critical path. Everything from Phase 2 onward can fan out across
contributors.

If only one user-visible milestone matters before anything else, it is:
**`generate("meta-llama/Llama-3.2-1B-Instruct", "Hello")` returning sensible
tokens on CPU.** Every phase here is in service of that, or of generalizing it.

---

## Guiding principles

These exist because `Transformers.jl` accumulated friction over six years.
Re-read this list whenever a PR feels like it's growing.

1. **Lean dependency tree.** Every dependency must be either (a) widely used
   in the Julia ecosystem (Flux, NNlib, SafeTensors, JSON3, HTTP) or (b)
   vendored inline. No more chains of single-maintainer helper packages
   (`FuncPipelines`, `BangBang`, `Pickle`, `NeuralAttentionlib`,
   `BytePairEncoding`, `DoubleArrayTries`, `Fetch`, `ValSplit`,
   `WordTokenizers`, `TextEncodeBase`, `StructWalk`, `PrimitiveOneHot`,
   `Tricks`) — most of these are abandoned alongside Transformers.jl itself.
2. **Safetensors only.** No Pickle, no PyTorch `.bin`. If a model isn't
   published in safetensors, we don't load it. The whole ecosystem has moved.
3. **Generic loaders, not per-model branches.** A new model adds a config +
   layer wiring + state-dict mapping. It does **not** add a tokenizer
   subclass, a textencoder subclass, or a special download path. The
   hardcoded `bert_textencoder` / `gpt_textencoder` / `t5_textencoder` split
   in the old project blocked half the modern model lineup; we don't repeat it.
4. **Decoder generation is a first-class feature, not an afterthought.**
   KV-cache, sampling, and chat templating land in Phase 1 — not Phase 4.
5. **Numeric parity is the test.** Every model has a CPU test that loads HF
   weights, runs a fixed input, and asserts logits match a recorded reference
   within tolerance. No parity test, no merge.
6. **No premature abstraction.** Three similar layers beat one configurable
   meta-layer. We can refactor once five exist.
7. **Modern Flux/Functors/Zygote from day one.** Track current minors, don't
   carry legacy version compat.

---

## Phase 0 — Bootstrap

Goal: an empty but installable package, CI green, with a North Star
example that prints `TODO` until Phase 1 fills it in.

- [x] `Project.toml` with the lean dependency set (Flux, NNlib, SafeTensors,
      JSON3, HTTP, Downloads, SHA, FillArrays, ChainRulesCore). Pin to
      Julia ≥ 1.10, test on 1.10 and 1.12.
- [x] `src/Allspark.jl` skeleton with submodule layout: `HFHub`, `Tokenizers`,
      `Layers`, `Models`, `Generation`.
- [x] GitHub Actions: test matrix (Julia 1.10/1.12 × Linux/macOS), Aqua.jl,
      JET.jl smoke pass, formatter check.
- [x] GPU CI plan documented (Buildkite or self-hosted CUDA runner). Wire it
      in Phase 1; without it Phase 4 is unsafe to merge.
- [x] `README.md` with the *aspirational* example (a `generate(...)` call)
      and a clear "status: pre-alpha" banner.
- [x] CONTRIBUTING.md describing the "config + layers + state-dict map +
      parity test + example" five-part contract for adding a model.

## Phase 1 — Core path: load a Llama, run `generate`

This is the smallest end-to-end vertical slice. Everything is built to serve
this one path; we generalize afterward.

- [x] **HFHub client.** Resolve `org/repo[@revision]`, download
      `config.json` + `tokenizer.json` + `model.safetensors` (and shard
      index) via `HTTP.jl` + stdlib `Downloads`. Cache to
      `~/.cache/huggingface/hub` matching Python's layout so users share
      downloads across stacks.
- [x] **Safetensors loader.** Lazy `mmap`-backed reader returning
      `Dict{String, AbstractArray}` keyed on HF parameter names. Handle
      sharded safetensors (`model.safetensors.index.json`) on day one.
- [x] **Generic `tokenizer.json` parser.** Implement the Rust-tokenizer JSON
      schema (BPE + byte-level pretokenizer is enough for Llama/GPT2/Qwen).
      No dependency on `BytePairEncoding.jl`. Tokenize/detokenize round-trips
      must match HF byte-for-byte for a fixture set.
- [x] **Minimal layer kit.** `RMSNorm`, `RoPE` (with scaling hooks for
      Llama3), `Linear` (no bias variant), `SiLU`-gated MLP, GQA attention.
      All `Flux.@layer`-annotated, all `Functors.fmap`-friendly.
- [x] **`LlamaModel` + `LlamaForCausalLM`.** Hardcoded for now; serves as the
      template for Phase 2's generalization.
- [x] **State-dict mapping.** Pure-data table mapping HF keys
      (`model.layers.0.self_attn.q_proj.weight`) to internal layer addresses.
      Loader walks the table; no method dispatch tricks.
- [x] **KV-cache.** Preallocated `(head_dim, n_kv_heads, max_seq, batch)`
      tensors (feature-first Julia layout — deviates from the originally
      planned PyTorch ordering so cache writes/reads are zero-copy slice
      assignments instead of `permutedims`), mutated in place per step.
      One shared `KVCache` type, threaded through every decoder block as
      a plain value — not via global state or `BangBang`-style functional
      updates. `build_caches(lm, max_seq, batch)` allocates a per-layer
      bundle; `reset!(caches)` zeros them for reuse.
- [x] **`generate()` API.** Greedy, temperature, top-k, top-p, repetition
      penalty, `max_new_tokens`, EOS stopping. Signature should feel familiar
      to anyone coming from HF's `model.generate`.
- [x] **Chat templating.** Minimal Jinja subset (`if` / `elif` / `else`,
      `for`, `set`, variables, attribute + item access, slicing,
      arithmetic / comparison / logical ops, `in` / `not in`,
      `is defined`, filters: `trim`, `tojson`, `raise_exception()`, loop
      variables, whitespace control) — enough to render simplified
      Llama-3, Qwen2.5, and Mistral templates from `tokenizer_config.json`.
      No full Jinja engine (no tuples / list / dict literals, no float
      literals, no macros, no includes, no custom filters or tests).
- [x] **REPL chat example.** `examples/repl_chat.jl` downloads a
      Llama via `HFHub.snapshot_download`, parses `config.json` /
      `tokenizer_config.json` / `generation_config.json` inline so the
      reader sees the full HF → Allspark mapping, builds
      `LlamaForCausalLM` + `load_state_dict!`, then loops on `stdin`
      applying the chat template per turn. Falls back to a hardcoded
      Llama-3 template when the model's bundled `chat_template` uses
      Jinja features outside our scope (tool calls, date filters,
      tuple literals).
- [x] **First parity test.** `test/parity_llama.jl` loads a Llama
      variant, encodes a fixed prompt, runs the model in fp32 CPU,
      and asserts that the last-position argmax + top-50 logits match
      a recorded HuggingFace fixture within `1e-3` (per-fixture
      `tolerance` overridable). Variants are individually gated so a
      laptop can run 1B / 3B while a server opts into 8B / 70B; see
      Phase 2's "Llama 3.x variants" entry for the full table.

## Phase 2 — Generalize: model coverage

Each model needs the five-part contract: config struct, layer wiring,
state-dict map, numeric-parity test, runnable example. Bias toward modern
decoder LLMs, since that's where demand lives.

- [x] Llama 3.x variants. `test/parity_llama.jl` is parameterized over
      a `VARIANTS` table — `1B` (`meta-llama/Llama-3.2-1B-Instruct`),
      `3B` (`Llama-3.2-3B-Instruct`), `8B` (`Llama-3.1-8B-Instruct`),
      `70B` (`Llama-3.1-70B-Instruct`). Each variant runs only when
      both (a) the env gate `ALLSPARK_TEST_PARITY` names it (or `all`)
      and (b) the corresponding fixture file exists, so a laptop run
      stops at 1B / 3B and a server can opt into 8B / 70B without
      changing code. Fixtures are recorded once by
      `python3 test/fixtures/record_llama_parity.py <VARIANT>`.
- [x] Mistral. `MistralConfig` + `MistralForCausalLM` in
      `src/Models/mistral.jl` reuse `LlamaDecoderLayer` / `LlamaModel`
      (the generic decoder pieces) and thread `cfg.sliding_window`
      into `GQA`. `GQA` grew a `window_size::Union{Nothing,Int}` field
      and a corresponding mask branch that fires both during prefill
      (when `seq_len > 1`) and during cached decode (when
      `seq_len_kv > window`). The HF parameter naming is identical to
      Llama's, so `mistral_state_dict_map` delegates to the shared
      `_decoder_state_dict_map` helper. Parity is gated on
      `ALLSPARK_TEST_PARITY_MISTRAL` (separate from the Llama gate)
      and covers v0.2 (sliding_window=4096) and v0.3 (no sliding) via
      `python3 test/fixtures/record_mistral_parity.py <VARIANT>`.
- [ ] Qwen2 / Qwen2.5 — tied embeddings, different RoPE base.
- [ ] Gemma2 — logit softcap, sliding window, attention scaling.
- [ ] Phi-3.
- [ ] GPT-2 / GPT-NeoX (encoder-light legacy support, useful for tests and
      embedding workflows).
- [ ] BERT / RoBERTa — encoder path, for embeddings & classification.
- [ ] **Refactor checkpoint.** Once five models exist, factor out the common
      `DecoderBlock` shape. Not before.
- [ ] Mixtral / MoE — deferred until a routing abstraction has at least one
      consumer.

## Phase 3 — Tokenizer breadth

`tokenizer.json` covers most modern models, but a few still need
SentencePiece or WordPiece directly.

- [ ] SentencePiece path for Gemma/T5-family. Either bind to
      `sentencepiece` via JLL or port a minimal Unigram decoder. Prefer the
      latter to keep deps lean if feasible.
- [ ] WordPiece path for legacy BERT checkpoints that ship `vocab.txt`
      instead of `tokenizer.json`.
- [ ] Tokenizer parity test harness: random sample of `tokenizer_config.json`
      fixtures, asserted against `transformers` Python output.

## Phase 4 — Performance & deployment

- [ ] fp16 / bf16 inference with documented numerical tolerance per model.
- [ ] FlashAttention kernel path (CUDA via `CUDA.jl` extension; ROCm via
      `AMDGPU.jl` extension). Fall back to fused-softmax-of-matmul on CPU.
- [ ] Int8 weight-only quantization on load.
- [ ] **GGUF read support.** This single feature opens the entire local-LLM
      ecosystem (llama.cpp checkpoints, community quantizations).
- [ ] Speculative decoding scaffolding (draft model + target model).
- [ ] Continuous batching for server-style deployment (defer until a real
      consumer asks for it).

## Phase 5 — Beyond text

- [ ] Embedding models (BGE, E5, nomic-embed). Mostly BERT variants — easy
      wins, high demand.
- [ ] ViT, then SigLIP, then DINOv2.
- [ ] Whisper.
- [ ] LLaVA / Llama-3.2-Vision once vision encoders + decoder LLMs are
      both healthy.

## Phase 6 — Fine-tuning

- [ ] LoRA / adapter load + save (PEFT-format compatible).
- [ ] Gradient checkpointing.
- [ ] Thin trainer example (Flux training loop in `examples/`, not a
      framework). Don't reinvent `transformers.Trainer`.

---

## Cross-cutting work (every phase)

- [ ] Doc page + runnable example per model.
- [ ] CPU numeric-parity test per model, recorded against HF Python output.
- [ ] `good first issue` tags on each state-dict mapping task — they are
      ideal on-ramps.
- [ ] Changelog entry per merged PR; we don't repeat the Transformers.jl
      mistake of going dark for an entire minor line.

## Explicit non-goals

- TensorFlow / JAX / ONNX import.
- Pickle / `.bin` checkpoint support.
- A full Jinja engine. We support the subset HF chat templates actually use.
- Reimplementing every HF `pipeline`. Users compose `tokenizer` + `model` +
  `generate` themselves; we are a library, not a framework.
- Backwards compatibility with `Transformers.jl` APIs. We learn from it, we
  don't ship a shim layer.
