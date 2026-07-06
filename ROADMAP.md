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
- [x] Mixtral / MoE (`src/Models/mixtral.jl`). Sparse
      Mixture-of-Experts decoder. The decoder block shape is Mistral's
      exactly — RMSNorm + GQA + RoPE + optional sliding window — so
      `DecoderLayer` is reused unchanged. The only new piece is the
      `MoEMLP` layer primitive: a bias-less `Linear(hidden,
      num_experts)` gate scores each token, the per-token top-K
      experts are softmax-renormalized, and the output is the
      weighted sum of those experts' `SiLUGatedMLP` outputs. The
      forward batches tokens per expert (each expert runs once with
      its assigned slice) but doesn't fuse the expert matmuls — Phase 4.
      State-dict naming follows HF's `block_sparse_moe.gate.weight`
      and `block_sparse_moe.experts.{i}.{w1,w2,w3}.weight`, where
      Mixtral's `w1`/`w2`/`w3` map to our standard
      `gate_proj`/`down_proj`/`up_proj` (yes, `w2` is the
      down-projection — non-obvious). `tie_word_embeddings=false` on
      every shipped Mixtral checkpoint. Parity is gated on
      `ALLSPARK_TEST_PARITY_MIXTRAL` over `8x7b` / `8x22b` via
      `python3 test/fixtures/record_mixtral_parity.py <VARIANT>` —
      both are server-class in fp32 (~90 GB / ~280 GB). Example:
      `examples/repl_chat_mixtral.jl` (uses Mistral's `[INST]` chat
      template since Mixtral shares the format).

## Phase 3 — Tokenizer breadth

`tokenizer.json` covers most modern models, but a few still need
SentencePiece or WordPiece directly.

- [x] SentencePiece path for Gemma/T5-family. Ported a minimal
      Unigram tokenizer in-tree (no `sentencepiece` JLL dependency).
      `src/Tokenizers/unigram.jl` builds a byte-keyed trie from the
      vocab and runs best-segmentation Viterbi to find the
      highest-log-probability tokenization. `byte_fallback=true`
      (Gemma's default) adds 256 single-byte fallback transitions
      using the `<0xHH>` tokens for out-of-vocab characters
      (multi-byte UTF-8 chars split into per-byte tokens). The
      Metaspace pretokenizer (space → `▁`, optional prepend) and the
      Replace / ByteFallback / Fuse / Strip decoders are all wired into
      the `tokenizer.json` parser, so a Gemma `tokenizer.json` loads
      and round-trips end-to-end. `Tokenizer` is now parameterized on
      the model type (`Tokenizer{M}`) and `encode_word(model, …)`
      dispatches between BPE and Unigram; existing BPE callsites are
      backward-compatible. Tested in `test/tokenizers.jl` (Viterbi
      best-path correctness, byte fallback on out-of-vocab and
      multi-byte UTF-8, Metaspace prepend rules, full Gemma-shape
      decoder sequence round-trip including `café` byte-fallback
      reassembly).
- [x] WordPiece path for the BERT family. `src/Tokenizers/wordpiece.jl`
      implements greedy longest-prefix-match `encode_word` with the
      `##` continuing-subword convention, falling back to `[UNK]` for
      unmatchable words or words longer than `max_input_chars_per_word`.
      Three new pieces land alongside: `BertPreTokenizer` (whitespace
      + punctuation split, matching BERT's `BasicTokenizer`),
      `WordPieceDecoder` (strip `##`, fold punctuation spacing,
      handle the `n't` / `'s` contractions), and a `Normalizer`
      abstraction with `BertNormalizer` (NFD-decompose + drop
      combining marks via `\\p{Mn}`, lowercase, CJK char padding,
      control-char cleanup) wired into `Tokenizer` as a new
      `normalizer` field that runs before pretokenization. `decode`
      grew a per-decoder token separator (`""` for ByteLevel-style
      pipelines, `" "` for WordPiece) so the WordPiece decoder can
      see the `" ##"` joins. `load_wordpiece_from_vocab_txt(path;
      lowercase, strip_accents)` is the legacy entry point for
      checkpoints that ship `vocab.txt` only — line number is the
      0-indexed token ID, and the loader builds a BERT-shaped
      tokenizer with sensible defaults. The `tokenizer.json` parser
      now recognizes `"WordPiece"` model + decoder, `"BertNormalizer"`
      (including the `Sequence`-of-normalizers wrapper), and
      `"BertPreTokenizer"`. Added `Unicode` to `Project.toml` for
      `Unicode.normalize`. Tests cover greedy longest-match,
      continuing-prefix lookup, max-length UNK fallback, normalizer
      lower+accent combinations, control-char cleanup, CJK padding,
      decoder cleanup of `don ' t` → `don't` and `do n't` → `don't`,
      and an end-to-end vocab.txt round-trip.
- [x] Tokenizer parity test harness. `test/parity_tokenizer.jl` is a
      family-agnostic gate that walks a `VARIANTS` table of 10
      tokenizers spanning all three families: byte-level BPE (gpt2,
      roberta, qwen2.5, pythia, phi-3, llama3.2, mistral-v0.2),
      SentencePiece-Unigram with byte fallback (gemma-2), and
      WordPiece (bert-base-uncased, bert-base-cased). Each variant
      reads a fixture recorded once by
      `test/fixtures/record_tokenizer_parity.py`, which captures HF's
      `encode(text, add_special_tokens=False)` IDs *and* the round-trip
      decoded string for 17 prompts per tokenizer (ASCII, accents, CJK,
      Korean, emoji, whitespace edges, punctuation salad, contractions,
      empty string, single char). The Julia side downloads the
      tokenizer via `snapshot_download`, loads it with `load_tokenizer`,
      and asserts both `encode(tk, text) == expected_ids` and
      `decode(tk, expected_ids) == expected_decoded`. Gated on
      `ALLSPARK_TEST_PARITY_TOKENIZER` (`all` / comma-separated subset
      / single label); unknown variants fail loud with the full valid
      list. Missing fixtures skip cleanly with the exact recorder
      command. The harness pairs naturally with the existing model
      parity tests — together they verify both stages of the
      tokenize→model pipeline match HF byte-for-byte.

## Phase 4 — Performance & deployment

- [x] fp16 / bf16 inference. `src/Models/dtype.jl` adds
      `convert_eltype(model, T)` which walks the model with
      `Flux.fmap` and converts every Float-array and Float-scalar leaf
      (so `RMSNorm.eps`, `RoPE.inv_freq`, every weight, every bias)
      to the target element type. `fp16` / `bf16` / `fp32` are
      one-liner wrappers; the conversion is pure-functional (the
      input model is unchanged). Allocate matching KV caches with
      `build_caches(lm, max_seq, batch; eltype=Float16)` etc.
      Added `BFloat16s` to `Project.toml` since BFloat16 isn't in
      Base. The docstring on `convert_eltype` documents the per-dtype
      tolerance band: `Float32 < 1e-3` (the recorded reference),
      `BFloat16 ≈ 1e-1` to `5e-1`, `Float16 ≈ 5e-2` to `5e-1`. Deeper
      models accumulate more error. Per-model fp16/bf16 parity is
      empirical — re-record a fixture with `torch_dtype=torch.float16`
      and compare against an `fp16(model)` run. Known limitation: the
      `Float32(-1e9)` mask constant and the Gemma softcap divisor in
      GQA promote attention activations back to Float32 transiently
      even when weights are lower precision; inference is correct but
      doesn't realize the full throughput advantage of pure fp16. A
      Phase 4 perf pass will widen those constants to be
      eltype-aware.

      Also: JET smoke is gated on `ALLSPARK_TEST_JET=1`. JET 0.11+
      currently fails to precompile on Julia 1.12.6 (upstream
      LoweredCodeUtils method-signature drift); the test is preserved
      for when a compatible JET tag lands but won't fail
      `Pkg.test()` in the meantime.
- [x] FlashAttention kernel path. `sdpa(q, k_t, v; scale, softcap, drop)`
      in `src/Layers/Layers.jl` holds the attention core that used to live
      inline in `GQA`, and acts as the dispatch boundary. Its default
      method materializes the `(seq_q, seq_kv)` score block and fuses
      softmax with the value matmul. That method is the CPU fallback and
      the reference the model parity tests pin, so lifting it out changed
      no results. `flash_sdpa` implements the FlashAttention forward: a
      tiled online-softmax recurrence (running max, running denominator,
      running weighted-value accumulator) that streams the KV sequence in
      `block_size` chunks. The full score matrix never exists at once, so
      peak extra memory drops from `O(seq_q · seq_kv)` to
      `O(seq_q · block_size)`. Allocations go through `similar` and the
      accumulator runs in `Float32`, so the same code runs on CPU and on
      device arrays. CPU tests check it against the materialized path
      within `~1e-4` on causal, sliding-window, softcap, and multi-block
      inputs, confirm finite zeros (not NaN) for a fully-masked query, and
      confirm fp16-in gives fp16-out. The CUDA (`AllsparkCUDAExt`), AMDGPU
      (`AllsparkAMDGPUExt`), and Metal (`AllsparkMetalExt`) extensions live
      under `ext/` (matching the `[extensions]` block in `Project.toml`,
      which had no files until now). Each adds an `sdpa` method on its
      device array type that moves the host-built drop mask to the device
      and calls `flash_sdpa`. Verified on an RTX 5090 (Blackwell, sm_120,
      CUDA.jl auto-resolved a compatible runtime): `test/gpu_attention.jl`
      runs the extension dispatch on real `CuArray`s and matches the CPU
      path within `1e-3` across no-mask, causal, and softcap+sliding-window
      cases. Scope stops at the algorithm plus the dispatch plumbing; a
      hand-fused single-CUDA-kernel flash (the reference CUTLASS-style
      kernel) is future work. The device speedup here comes from on-GPU
      `batched_mul`/`softmax` and the tiling's memory bound, not a custom
      kernel.
- [x] Int8 weight-only quantization. `QuantizedInt8Matrix` (in
      `src/Layers/Layers.jl`) stores a weight as per-output-row symmetric
      int8 (`scale[i] = maximum(abs, W[i,:]) / 127`) and subtypes
      `AbstractMatrix{Float32}`, so its `*` dequantizes before the matmul
      and `Linear` needs no forward change. `quantize_int8(model)`
      (`src/Models/quantize.jl`) walks the model with `fmap` and swaps each
      `Linear` weight for a `QuantizedInt8Matrix`, leaving norms and token
      embeddings in Float32; it mirrors the `fp16`/`bf16` post-load
      wrappers. Weight-only and correctness-first: the dequant is
      materialized per call (no int8 GEMM kernel on CPU/GPU), so the win is
      ~4x smaller resident weights, not speed. Logit error vs fp32 runs
      `1e-2`–`1e-1`. Tested in `test/quantize.jl` (round-trip within one
      quantization step, matmul matches the dequantized weight, Linears
      quantized while embeddings/norms stay Float32, forward + KV-cache
      decode track fp32, Qwen's biased QKV handled).
- [x] GGUF read support. `src/GGUF/GGUF.jl` is a new submodule whose
      `load_gguf(path)` parses the GGUF v3 container: magic + version, the
      full metadata key/value table (all 13 value types including nested
      arrays), and the tensor info table (name, dims, GGML dtype, offset),
      honouring `general.alignment` (default 32). It returns a `GGUFFile`
      with `version`, `metadata`, `tensors` (dequantized to `Float32`), and
      `tensor_types`. Tensor dims keep GGML's contiguous-first order, which
      maps straight onto Julia's column-major layout (mapping to HF's
      row-major `(out, in)` weights is a transpose left to the model-wiring
      step). Dequant covers F32, F16, Q8_0, and Q4_0 — the unquantized and
      the two simplest block-quant formats, which between them cover most
      community checkpoints. The k-quants (Q4_K/Q5_K/Q6_K and friends) have
      intricate superblock layouts and are not dequantized yet; hitting one
      throws an `ArgumentError` naming the dtype rather than returning
      garbage. Tested in `test/gguf.jl` via a Julia-side GGUF writer (so
      round-trips need no external fixture): metadata of every exercised
      type round-trips, F32 is exact, F16 within Float16 precision, Q8_0 and
      Q4_0 within one quantization step, an unsupported dtype and a bad magic
      both error. Next step to actually run a llama.cpp checkpoint is the
      GGUF→model mapping (architecture/config from metadata, `blk.N.*` tensor
      names → Allspark's HF-style names); that's mechanical but wants real
      fixtures to verify, so it's its own chunk.
- [x] Speculative decoding scaffolding (draft model + target model).
      `speculative_generate(target, draft, input_ids; n_draft, …)` in
      `src/Generation/speculative.jl`. Each round: the small `draft` model
      proposes `n_draft` (γ) tokens autoregressively, the `target` verifies all
      γ in a single forward pass, and the Leviathan/Chen accept-reject rule
      keeps the output distributed exactly as `target` alone. Greedy
      (`do_sample=false`) accepts draft token `i` iff it equals
      `argmax(target_i)` and on the first miss emits the target's argmax;
      sampling (`do_sample=true`, `temperature`) accepts with probability
      `min(1, p_i/q_i)`, resamples a rejection from `normalize((p_i-q_i)₊)`,
      and draws a bonus token from the target's last-position distribution when
      all γ are accepted. The two-model KV-cache rollback is handled by the
      invariant "both caches valid for positions 1..L-1, last committed token
      unfed": each round sync-feeds that token to both models (the one
      single-token pass that can't be avoided since the target distribution
      after the final committed token was never computed), feeds all γ drafted
      tokens into the draft cache so it stays valid through an accept-all round,
      and lets the next round overwrite stale slots beyond the accepted length
      (attention only ever reads `1..step+seq-1`, so stale tail entries are
      never seen). Draft and target must share a vocabulary; a mismatched
      LM-head width raises `DimensionMismatch`. Correctness-first scaffolding:
      `top_k`/`top_p`/`repetition_penalty` aren't applied in the speculative
      path yet, and the per-round sync feed is recompute a tuned implementation
      would fold into the next verify pass. Tested in `test/speculative.jl`
      (50 assertions): the load-bearing one is that greedy speculative
      reproduces `generate(target, …)` byte-for-byte across `n_draft ∈
      {1,2,4,8}` and several `max_new_tokens`, for both a *different* draft
      (forces rejections + cache rollback) and an *identical* draft (forces
      the accept-all + bonus path) — which validates the accept/reject math and
      the dual-cache bookkeeping together, independent of draft quality. Plus
      exact `max_new_tokens` capping, EOS, sampling reproducibility, and the
      vocab-mismatch guard.
- [ ] Continuous batching for server-style deployment (defer until a real
      consumer asks for it).

## Phase 5 — Beyond text

- [x] Embedding models (BGE, E5). `BertEmbeddingModel`
      (`src/Models/embedding.jl`) wraps the `BertModel` trunk with a pooling
      strategy (`:cls` for BGE, `:mean` for E5) and optional L2 normalization
      (default on; retrieval checkpoints expect unit-norm vectors).
      `embed(model, ids)` returns one pooled vector; `embed(model, tokenizer,
      text|texts)` tokenizes first and leaves special tokens and instruction
      prefixes (E5's `query:`/`passage:`) to the caller. It exports the pooling
      helpers `mean_pool`/`cls_pool`/`l2_normalize`. Batch-1 encodes each text
      on its own, so `:mean` matches sentence-transformers' attention-masked
      mean with no padding to exclude, and the absent per-sequence padding mask
      never bites. Loading reuses `bert_state_dict_map` filtered to the trunk
      keys and ignores any pooler or MLM-head weights; that map now drops the
      separator on an empty prefix, so prefix-less BGE/E5 checkpoints (keys like
      `embeddings.word_embeddings.weight`) load. A shared `BertModel(cfg)`
      constructor came out of `BertForMaskedLM(cfg)`. `test/embedding.jl` runs
      27 assertions: pooling primitives, unit-norm output, `embed(ids)` against
      a manual pooled-and-normalized trunk pass, `:cls` versus `:mean` under
      shared weights, the prefix-less load path, and the tokenizer overloads.
      The gated `test/parity_embedding.jl` (plus `record_embedding_parity.py`)
      checks bge-small and e5-small against sentence-transformers within `1e-3`.
- [x] nomic-embed. `NomicBertConfig` / `NomicBertModel` (`src/Models/nomic.jl`)
      port the NomicBERT encoder. It keeps BERT's post-norm block
      (`x = norm1(x + attn(x)); x = norm2(x + mlp(x))`) and swaps three pieces:
      full rotary at `rope_theta=1000` (not Llama's 10000) replaces the absolute
      position table, a fused bias-free `Wqkv` feeds bidirectional MHA, and the
      MLP is gated SwiGLU. The reused layers do the work: `GQA(...; causal=false,
      qkv_bias=false, wo_bias=false)` with `RoPE(head_dim; base=rope_theta)` for
      attention and `SiLUGatedMLP` for the MLP (the gated keys map `fc12` → gate,
      `fc11` → up, `fc2` → down, so `fc11 * silu(fc12)` lands as `up *
      silu(gate)`). `NomicBertModel` plugs into the existing `BertEmbeddingModel`
      wrapper, which is generic over its trunk; `BertEmbeddingModel(cfg)` for a
      `NomicBertConfig` defaults to mean pooling (nomic's convention). Loading
      runs `nomic_state_dict_map` (embeddings transpose; everything else as-is)
      and then splits the fused `Wqkv` rows `[Q; K; V]` into `wq`/`wk`/`wv` by
      hand, since the generic loader has no unstack transform. Architecture
      details verified against the published `config.json` and
      `modeling_hf_nomic_bert.py`. Tested in `test/nomic.jl` (27 assertions:
      forward shape, bidirectional + rotary wiring, the `Wqkv` split and SwiGLU
      mapping, mean-pooled unit-norm output, `embed(ids)` against a manual pass)
      plus a gated `test/parity_nomic.jl` (+ `record_nomic_parity.py`) that
      checks nomic-embed-text-v1.5 against the HF model within `1e-3`.
- [x] ViT. `ViTConfig` / `ViTModel` / `ViTForImageClassification`
      (`src/Models/vit.jl`) port `google/vit-base-patch16-224`. It's a pre-norm
      encoder, so it reuses `DecoderLayer` with bidirectional `GQA` (`causal=false`,
      no rotary, biased QKV), exact-GELU `GeluMLP`, and `LayerNorm`. The one new
      component is patch embedding: HF's stride=kernel `Conv2d` over
      non-overlapping patches equals flattening each patch and applying a
      `Linear`, so `ViTPatchEmbeddings` reshapes the input to `(feat, patch,
      batch)` (channel-fastest features, width-patch-fastest order to match HF's
      `flatten(2)`) and the conv weight `(hidden, C, P, P)` reshapes to the
      `Linear` `(hidden, C*P*P)`. A learned `[CLS]` token and learned position
      embeddings are added, and the classifier reads the `[CLS]` position after a
      final LayerNorm. Attention is standard MHA with separate Q/K/V, so no
      unstacking is needed; `load_state_dict!` only reshapes the patch conv,
      squeezes `[CLS]` `(1,1,H)`, and transposes position embeddings `(1,S,H)`.
      The model takes a preprocessed `(C, H, W, batch)` pixel tensor; image
      decode/resize stays out of scope (no image deps), so a turnkey
      image→logits path is future work. Architecture verified against the
      published `config.json` and `modeling_vit.py`. Tested in `test/vit.jl` (32
      assertions: the load-bearing one checks patch embedding against an explicit
      nested-loop conv reference, pinning the layout; plus forward shape, pre-norm
      wiring, and the full load path) with a gated `test/parity_vit.jl` (+
      `record_vit_parity.py`) that feeds a saved `pixel_values` tensor and checks
      logits against HF within `1e-3`.
- [x] SigLIP. `SiglipConfig` / `SiglipVisionModel` / `SiglipTextModel` /
      `SiglipModel` (`src/Models/siglip.jl`) port `google/siglip-base-patch16-224`.
      Both towers are pre-norm encoders reusing `DecoderLayer` with bidirectional
      biased-QKV `GQA` and tanh-GELU `GeluMLP` (`gelu_pytorch_tanh`) at
      `layer_norm_eps=1e-6`. The vision tower has no `[CLS]` token: it reuses
      `ViTPatchEmbeddings`, adds a learned position table, runs the encoder +
      `post_layernorm`, then pools with `SiglipAttentionPoolingHead` — a single
      learned probe query cross-attending over the patch tokens (HF's
      `nn.MultiheadAttention`, whose fused `in_proj` splits into `wq`/`wk`/`wv` at
      load), followed by `probe_attn + mlp(layernorm(·))`. The text tower embeds
      tokens + positions, runs the encoder + `final_layer_norm`, pools the last
      token, and applies a linear `head`. `SiglipModel` L2-normalizes both
      embeddings and returns `logits_per_image = imgᵀ·txt · exp(logit_scale) +
      logit_bias`. Architecture verified against the published `config.json` and
      `modeling_siglip.py`. Tested in `test/siglip.jl` (45 assertions: the
      load-bearing one checks the pooling head against an explicit single-query
      multi-head reference; plus forward shapes, no-CLS wiring, the `in_proj`
      split, the patch/embedding reshapes, the `logit_bias` shift, and unit-norm
      embeddings) with a gated `test/parity_siglip.jl` (+ `record_siglip_parity.py`)
      that feeds a saved pixel tensor and padded text ids and checks
      `logits_per_image` against HF within `1e-2`.
- [ ] DINOv2. ViT-family image encoder reusing the patch embedding and pre-norm
      block; adds register tokens, LayerScale on each residual branch, and a
      SwiGLU MLP in the giant variant — its own head, so a separate chunk.
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
