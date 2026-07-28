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
