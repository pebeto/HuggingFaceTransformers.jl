#!/usr/bin/env python3
"""Record reference tokenizations for HuggingFaceTransformers.jl's GGUF tokenizer test.

Usage:
    python3 test/fixtures/record_gguf_tokenizer_parity.py

Requirements:
    pip install tokenizers huggingface_hub

A GGUF file carries its tokenizer in metadata, which the Julia loader rebuilds.
The reference is the tokenizer of the repository the GGUF was converted from:
its `tokenizer.json`, run through HF `tokenizers`, with the special tokens its
post-processor adds. Only the `tokenizer.json` files download. The strings cover
what tokenizers differ on: runs of spaces, digits, accents, CJK, emoji, tabs and
newlines, contractions, code, and the special tokens of several families.
"""
import json
import os

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# Variant name -> (GGUF repository, GGUF file, repository it was converted from).
VARIANTS = {
    "smollm2-q4_k_m": ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q4_K_M.gguf",
                       "HuggingFaceTB/SmolLM2-135M-Instruct"),
    "qwen2.5-0.5b-q4_k_m": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                            "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen3-0.6b-q2_k": ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q2_K.gguf", "Qwen/Qwen3-0.6B"),
    "gemma2-2b-q4_k_m": ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf", "unsloth/gemma-2-2b-it"),
    # Older conversions of TinyLlama (TheBloke's among them) wrote every score as 0,
    # which loses SentencePiece's merge order; this one keeps them.
    "tinyllama-1.1b-q2_k": ("mradermacher/TinyLlama-1.1B-Chat-v1.0-GGUF", "TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf",
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    "llama3.2-1b-q4_k_m": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                           "unsloth/Llama-3.2-1B-Instruct"),
}

TEXTS = [
    "Hello world! How are you today?",
    "  Leading spaces,   runs of   spaces, and trailing  ",
    "Numbers: 1234567 and 3.14159, plus 2026-11-19.",
    "café naïve façade Ñandú",
    "日本語のテキストと中文混合 한국어",
    "Emoji 😀🚀 and symbols ∑∫√ €£¥",
    "Tabs\tand\nnew\nlines\n\n\nend",
    "Don't won't I'm they're we'll it's",
    "def f(x):\n    return x ** 2  # square\n",
    "<|im_start|>user\nHi<|im_end|>\n<start_of_turn>model",
]

OUT = os.path.join(os.path.dirname(__file__), "gguf_tokenizer_parity.json")


def main():
    fixture = {"texts": TEXTS, "variants": {}}
    for name, (gguf_repo, gguf_file, repo) in VARIANTS.items():
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        ids = [tok.encode(t).ids for t in TEXTS]
        fixture["variants"][name] = {
            "repo_id": gguf_repo,
            "gguf_file": gguf_file,
            "tokenizer_repo": repo,
            "ids": ids,
            "decoded": [tok.decode(i, skip_special_tokens=False) for i in ids],
        }
        print(f"{name}: {sum(map(len, ids))} tokens")
    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
