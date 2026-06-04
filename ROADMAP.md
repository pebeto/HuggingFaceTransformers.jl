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
      `src/Models/mistral.jl` reuse `DecoderLayer` / `DecoderModel`
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
- [x] Qwen2 / Qwen2.5. `QwenConfig` + `QwenForCausalLM` in
      `src/Models/qwen.jl` reuse `DecoderLayer` / `DecoderModel`.
      The architectural deltas vs Llama/Mistral that landed here:
      `Linear` now carries an optional bias (dispatched on the type
      parameter so the no-bias path stays branch-free); `GQA` takes
      `qkv_bias=true` to wire bias into Q/K/V but not O; and
      `_decoder_state_dict_map` accepts `qkv_bias=true` to emit the
      `q_proj.bias` / `k_proj.bias` / `v_proj.bias` entries that
      Qwen weights carry. `tie_word_embeddings` is honored as for
      Llama (Qwen2.5 0.5B and 1.5B tie; 3B and 7B don't). Parity is
      gated on `ALLSPARK_TEST_PARITY_QWEN` and covers 0.5B / 1.5B /
      3B / 7B via `python3 test/fixtures/record_qwen_parity.py
      <VARIANT>`. Example: `examples/repl_chat_qwen.jl` with a
      ChatML-format fallback template.
- [x] Gemma2. Biggest model-shape delta in Phase 2: introduced new
      layer primitives and a new decoder block instead of reusing
      Llama's. The layer kit grew `GemmaRMSNorm` (uses `(1 + weight)`
      scaling), `GeluGatedMLP` (GELU tanh approximation as the MLP
      activation), and a `softcap(x, cap)` helper. `GQA` gained
      `softcap` and `query_scale` optional fields so Gemma2's
      attention-score softcap and per-variant `query_pre_attn_scalar`
      (2B=224, 9B=256, 27B=144) flow through without forking the
      attention layer. `GemmaDecoderLayer` is its own struct because
      Gemma's block has four RMSNorms per layer (pre + post for both
      attention and MLP) instead of Llama's two. `GemmaModel` also
      scales the input embedding by `sqrt(hidden_size)` (HF parity).
      The final logit softcap lives in `GemmaForCausalLM`'s forward.
      Sliding-window attention alternates per layer: even-indexed
      layers (0, 2, …) get `cfg.sliding_window`, odd-indexed get full
      causal. `tie_word_embeddings=true` on every Gemma2 checkpoint,
      handled identically to Llama's tied path. Parity is gated on
      `ALLSPARK_TEST_PARITY_GEMMA` over `2B` / `9B` / `27B` via
      `python3 test/fixtures/record_gemma_parity.py <VARIANT>`.
      Example: `examples/repl_chat_gemma.jl` with a
      `<start_of_turn>` / `<end_of_turn>` fallback template.
- [x] Phi-3. The fifth model — and the trigger for the
      Refactor checkpoint below. `Phi3Config` + `Phi3ForCausalLM` in
      `src/Models/phi3.jl` reuse `DecoderModel` / `DecoderLayer` /
      `GQA` / `SiLUGatedMLP` unchanged. The interesting part is the
      state-dict loader: Phi-3 ships `self_attn.qkv_proj.weight`
      (concatenated `[Q; K; V]`) and `mlp.gate_up_proj.weight`
      (concatenated `[gate; up]`) as single fused tensors. Rather than
      introducing fused-projection layer variants, `load_state_dict!`
      slices each fused tensor into the standard `wq`/`wk`/`wv` and
      `gate_proj`/`up_proj` at load time. (The single-fused-matmul
      perf path is Phase 4 territory.) Only `partial_rotary_factor =
      1.0` is supported; the loader rejects other values with a clear
      error since `longrope` scaling and partial RoPE need their own
      implementation. Parity is gated on `ALLSPARK_TEST_PARITY_PHI3`
      and covers `mini-4k` (3.8B) and `medium-4k` (14B) via
      `python3 test/fixtures/record_phi3_parity.py <VARIANT>`.
      Example: `examples/repl_chat_phi3.jl` with a
      `<|user|>` / `<|assistant|>` / `<|end|>` fallback template.
- [x] GPT-2. First non-Llama-family architecture (`src/Models/gpt2.jl`):
      `LayerNorm` instead of `RMSNorm` (added to the layer kit with
      weight + bias), learned absolute position embeddings added to
      token embeddings in the trunk (no RoPE in the attention layer —
      `GQA`'s `rope` field was relaxed to `Union{Nothing,RoPE}`),
      `GeluMLP` for the non-gated FFN with tanh-approximation GELU.
      `GQA` grew `wo_bias::Bool=false` so GPT-2 can keep its `c_proj`
      bias while the Llama family stays bias-less. The state-dict
      loader handles HF's GPT-2 quirks: HF stores linear weights in
      Conv1D orientation `(in, out)` so the loader transposes them,
      and `c_attn.weight` / `c_attn.bias` are fused tensors sliced
      into `wq` / `wk` / `wv` (à la Phi-3). `GPT2Model` is its own
      struct (not `DecoderModel`) because the trunk adds position
      embeddings into token embeddings before the layer stack; the
      layers themselves reuse `DecoderLayer`. `tie_word_embeddings=true`
      is always honored. Parity is gated on `ALLSPARK_TEST_PARITY_GPT2`
      across 124M / 355M / 774M / 1558M via
      `python3 test/fixtures/record_gpt2_parity.py <VARIANT>`. Example:
      `examples/completion_gpt2.jl` (text completion rather than chat
      — GPT-2 isn't instruction-tuned).
- [x] GPT-NeoX / Pythia (`src/Models/neox.jl`). Three new things on
      top of GPT-2's primitives: (1) **partial RoPE** — `RoPE` grew a
      `rotary_dim::Union{Nothing,Int}` field so only the first
      `floor(head_dim * partial_rotary_factor)` channels of each head
      get rotated; the tail passes through unchanged. Pythia and
      GPT-NeoX-20B both use `rotary_pct=0.25`. (2) **parallel-residual
      decoder block** — `NeoXDecoderLayer` reads `x` once and adds
      both branch outputs back: `x = x + attn(ln_a(x)) + mlp(ln_m(x))`.
      `DecoderModel` is reused since its forward doesn't care about
      layer type. (3) **interleaved per-head fused QKV** —
      `query_key_value.weight` is laid out as
      `[Q_h; K_h; V_h; Q_{h+1}; …]` (not GPT-2's concatenated
      `[Q; K; V]`), so the loader walks heads and copies each
      `(hd, hidden)` slice into the right rows of `wq` / `wk` / `wv`.
      NeoX uses `nn.Linear` (not Conv1D), so no transpose for
      `attention.dense`, `dense_h_to_4h`, `dense_4h_to_h`. The LM head
      `embed_out.weight` lives outside the `gpt_neox.` namespace and
      is loaded as-is (no transpose). The constructor rejects
      odd-valued rotated dims since the rotation operates on pairs.
      Parity is gated on `ALLSPARK_TEST_PARITY_NEOX` across
      pythia-70m / 410m / 1b / 6.9b via
      `python3 test/fixtures/record_neox_parity.py <VARIANT>`.
      Example: `examples/completion_neox.jl` (Pythia is a base LM,
      not chat-tuned).
- [x] BERT / RoBERTa (`src/Models/bert.jl`). First encoder-only model
      — the biggest architectural delta in Phase 2. Bidirectional
      attention (`GQA` grew `causal::Bool=true`; BERT passes `false`),
      post-norm encoder block (`BertEncoderLayer`: norm AFTER each
      residual add), three summed embeddings (token + position +
      token-type) followed by a LayerNorm in `BertEmbeddings`, and a
      MaskedLM head with `dense → gelu → LayerNorm → tied_decoder + bias`.
      Exact GELU is wired through `GeluMLP`'s new `approx::Bool` flag
      (default `true` for GPT-2 compat; `false` for BERT, RoBERTa, and
      — correctness fix — GPT-NeoX, which had been using the tanh
      approximation by inheritance). `BertConfig` is shared between
      BERT and RoBERTa; `hf_prefix` selects the state-dict namespace
      (`bert.` or `roberta.`), `head_prefix` selects the head naming
      (`cls.predictions.…` or `lm_head.…`), and
      `position_embedding_offset` accounts for RoBERTa's
      `padding_idx + 1` convention. Parity is gated on
      `ALLSPARK_TEST_PARITY_BERT` over RoBERTa-base / RoBERTa-large
      via `python3 test/fixtures/record_bert_parity.py <VARIANT>`;
      the fixture records logits at an explicit `<mask>` position.
      BERT proper is **not** in parity coverage yet: it uses
      WordPiece tokenization, which is Phase 3 work. Example:
      `examples/mask_fill_roberta.jl` (top-K predictions for a user
      `<mask>`-containing sentence).
- [x] **Refactor checkpoint.** With five models in place (Llama,
      Mistral, Qwen, Gemma2, Phi-3), the shared decoder shape was
      lifted into `src/Models/decoder.jl`: `LlamaDecoderLayer` →
      `DecoderLayer`, `LlamaModel` → `DecoderModel`,
      `_decoder_state_dict_map` and `TokenEmbedding` moved alongside.
      `llama.jl` now contains only Llama-specific code (config,
      `LlamaForCausalLM`, factory, state-dict map delegate, loader).
      `mistral.jl` / `qwen.jl` / `phi3.jl` reference the new names
      verbatim. `gemma.jl` is unchanged — Gemma2's four-norm decoder
      block stays separate as its own `GemmaDecoderLayer` /
      `GemmaModel`. All 258 model + 79 layer + tokenizer/generation/
      hub/aqua/JET tests still pass after the rename.
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
