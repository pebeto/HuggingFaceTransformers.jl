#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's GGUF parity test.

Usage:
    python3 test/fixtures/record_gguf_parity.py [VARIANT ...]

With no arguments every variant is recorded. Each one is a GGUF file that
transformers loads with `gguf_file=`, dequantizing every tensor to float32 the
way the Julia loader does, so the logits are compared at fp32 tolerance even for
the quantized files.

Requirements:
    pip install transformers torch gguf accelerate

The variants cover the four GGUF architectures the loader maps (llama, qwen2,
qwen3, gemma2) and between them every tensor type it decodes: llama.cpp falls
back from the k-quants to Q5_0, Q5_1, Q8_0 and IQ4_NL for rows shorter than a
256-value block, which the small models have plenty of.

Resource cost: SmolLM2 and the Qwen files are 100-500 MB each; gemma-2-2b is a
1.7 GB download and about 11 GB of RAM once dequantized.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANTS = {
    "smollm2-q4_k_m": ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    "smollm2-q3_k_m": ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q3_K_M.gguf"),
    "qwen2.5-0.5b-q4_k_m": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "qwen3-0.6b-q2_k": ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q2_K.gguf"),
    "gemma2-2b-q4_k_m": ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf"),
}

PROMPT = "The capital of France is"
TOP_K = 50
TOLERANCE = 1e-3
FIXTURES_DIR = os.path.dirname(__file__)


def record(variant):
    repo_id, gguf_file = VARIANTS[variant]
    tokenizer = AutoTokenizer.from_pretrained(repo_id, gguf_file=gguf_file)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, gguf_file=gguf_file, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    encoded = tokenizer.encode(PROMPT)
    with torch.no_grad():
        logits = model(torch.tensor([encoded])).logits[0, -1, :].float().numpy()

    top = sorted(range(len(logits)), key=lambda i: -logits[i])[:TOP_K]
    fixture = {
        "repo_id": repo_id,
        "gguf_file": gguf_file,
        "prompt": PROMPT,
        "input_token_ids": encoded,
        "argmax_token_id": int(top[0]),
        "top_indices": [int(i) for i in top],
        "top_logits": [float(logits[i]) for i in top],
        "tolerance": TOLERANCE,
    }
    out_path = os.path.join(FIXTURES_DIR, f"gguf_{variant}_parity.json")
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"{variant}: argmax {top[0]} ({tokenizer.decode([top[0]])!r}), wrote {out_path}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variant(s) {unknown}; pick from {list(VARIANTS)}")
    for name in names:
        record(name)
