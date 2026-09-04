```@meta
CurrentModule = HuggingFaceTransformers
```

# Tokenizers

[`Tokenizers.load_tokenizer`](@ref) reads a HuggingFace `tokenizer.json` and
returns a [`Tokenizers.Tokenizer`](@ref). There are no per-model tokenizer
subclasses; the JSON drives everything.

```julia
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode, decode

tokenizer = load_tokenizer(dir)
ids = encode(tokenizer, "Hello, world!")
text = decode(tokenizer, ids)
```

## Special tokens

`encode` applies the checkpoint's `post_processor`, so the ids come back wrapped
the way the model expects: `[CLS] … [SEP]` for BERT, `<s> … </s>` for RoBERTa,
`<|startoftranscript|> <|notimestamps|> … <|endoftext|>` for Whisper. Byte-level
processors add nothing. This matches HF, where `add_special_tokens` defaults to
true, and it matters: a masked-LM or embedding checkpoint fed bare ids returns
noticeably worse results.

Pass `add_special_tokens = false` when the text already carries them, which is
the case for chat-template output, or when you need the model's own segmentation
alone:

```julia
prompt = apply_chat_template(template, messages; add_generation_prompt=true)
ids = encode(tokenizer, prompt; add_special_tokens=false)  # template emitted them
```

Only the single-sequence form is supported. Pair encoding
(`[CLS] A [SEP] B [SEP]`), padding, truncation, and offset mapping are not
implemented yet.

Three segmentation models are supported:

- BPE with ByteLevel pre-tokenization, for the GPT-2, Llama-3, Qwen2, and
  RoBERTa family.
- Unigram with Metaspace pre-tokenization, for SentencePiece checkpoints such as
  Gemma and T5.
- WordPiece with `BertNormalizer`, for BERT and its descendants.

Older checkpoints that ship only a `vocab.txt` and no `tokenizer.json` load
through [`Tokenizers.load_wordpiece_from_vocab_txt`](@ref), which treats the
0-indexed line number as the token ID and assembles a BERT-shaped tokenizer.

## Special tokens

A tokenizer's `added_tokens` table holds [`Tokenizers.AddedToken`](@ref)
entries, matched ahead of the model's own segmentation. Pass
`skip_special_tokens=true` to [`Tokenizers.decode`](@ref) to drop them from the
output:

```julia
decode(tokenizer, ids; skip_special_tokens=true)
```

## Parity with Python

Round-trip behavior is checked against `AutoTokenizer` output on a prompt set
shared across every tokenizer family, so failures are directly comparable
between them. The prompts cover accented Latin text, CJK, Korean, emoji, and
leading and trailing whitespace.

Recording a fixture needs `transformers` installed:

```
python3 test/fixtures/record_tokenizer_parity.py all
```

Then run the comparison, which is opt-in because it downloads real tokenizers:

```
HFT_TEST_PARITY_TOKENIZER=all julia --project -e 'using Pkg; Pkg.test()'
```

The variable takes a single label, a comma-separated list, or `all`. Unknown
labels fail with the valid set listed, and a missing fixture skips with the
exact recorder command to run.
