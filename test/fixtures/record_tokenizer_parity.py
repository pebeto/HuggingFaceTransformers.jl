#!/usr/bin/env python3
"""Record HF tokenizer encodings + decodings for the HuggingFaceTransformers.jl tokenizer
parity harness (`test/parity_tokenizer.jl`).

Usage:
    python3 test/fixtures/record_tokenizer_parity.py [VARIANT]
    python3 test/fixtures/record_tokenizer_parity.py all

VARIANT is one of the labels defined in `VARIANTS` below. `all` records
every variant; useful when wiring up CI.

Requirements:
    pip install transformers
    huggingface-cli login         # for gated repos (Llama, Gemma)

Each variant produces a JSON fixture with:
    {
      "repo_id": "...",
      "tokenizer_family": "bpe" | "unigram" | "wordpiece",
      "samples": [{"text": "...", "ids": [...], "decoded": "..."}, ...]
    }

The samples list deliberately spans:
- ASCII text (English, short and long).
- Accented Latin (`café`, `naïve`).
- CJK (`你好世界`).
- Korean (`한국어 입력`).
- Emoji (`🚀`).
- Whitespace edges (leading, trailing, repeated).
- Punctuation and contractions (`don't`, `1+2=3`).
- The empty string and a single character.

`encode` uses `add_special_tokens=False` so the IDs land 1:1 with what
HuggingFaceTransformers produces (HuggingFaceTransformers's encoder doesn't auto-prepend BOS).
`decode` uses `skip_special_tokens=False` and
`clean_up_tokenization_spaces=True` for tokenizers that support that
flag, matching HuggingFaceTransformers's defaults.
"""
import json
import os
import sys

from transformers import AutoTokenizer

VARIANTS = {
    # label              (repo_id, family)
    "gpt2":              ("gpt2",                                  "bpe"),
    "roberta":           ("FacebookAI/roberta-base",               "bpe"),
    "qwen2.5":           ("Qwen/Qwen2.5-0.5B-Instruct",            "bpe"),
    "pythia":            ("EleutherAI/pythia-70m",                 "bpe"),
    "phi-3":             ("microsoft/Phi-3-mini-4k-instruct",      "bpe"),
    "llama3.2":          ("meta-llama/Llama-3.2-1B-Instruct",      "bpe"),
    "mistral-v0.2":      ("mistralai/Mistral-7B-Instruct-v0.2",    "bpe"),
    "gemma-2":           ("google/gemma-2-2b-it",                  "unigram"),
    "bert-uncased":      ("google-bert/bert-base-uncased",         "wordpiece"),
    "bert-cased":        ("google-bert/bert-base-cased",           "wordpiece"),
}

# The prompt mix is shared across variants so the failure surface is
# directly comparable between tokenizer families.
SAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "1+2=3 and π ≈ 3.14159",
    "café au lait",                              # accents
    "naïve résumé über",                          # more accents
    "你好世界",                                    # CJK
    "한국어 입력",                                  # Korean
    "🚀 rocket emoji",                            # emoji
    "  leading whitespace",
    "trailing whitespace   ",
    "tabs\tand\nnewlines",
    "multiple   spaces    in    a    row",
    "don't can't won't shouldn't",               # contractions
    ".,!?:;-—()[]{}\"'",                          # punctuation salad
    "",                                          # empty
    "a",                                         # single char
    "The capital of France is",                  # the canonical parity prompt
]

FIXTURES_DIR = os.path.dirname(__file__)


def record_one(label):
    if label not in VARIANTS:
        print(f"Unknown variant: {label!r}. Valid: {list(VARIANTS)}", file=sys.stderr)
        sys.exit(1)

    repo_id, family = VARIANTS[label]
    out_path = os.path.join(
        FIXTURES_DIR, f"tokenizer_parity_{label.replace('.', '_').replace('-', '_')}.json"
    )

    print(f"Loading tokenizer for {repo_id} ({family})...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    samples = []
    for text in SAMPLES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        try:
            decoded = tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
        except TypeError:
            # Older tokenizers may not accept clean_up_tokenization_spaces.
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
        samples.append({"text": text, "ids": [int(i) for i in ids], "decoded": decoded})

    fixture = {
        "repo_id": repo_id,
        "tokenizer_family": family,
        "samples": samples,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(samples)} samples to {out_path}")


def main(argv):
    if len(argv) > 1 and argv[1] == "all":
        for label in VARIANTS:
            record_one(label)
    else:
        label = argv[1] if len(argv) > 1 else "gpt2"
        record_one(label)


if __name__ == "__main__":
    main(sys.argv)
