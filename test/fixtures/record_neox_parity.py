#!/usr/bin/env python3
"""Record reference logits for Allspark.jl's GPT-NeoX (Pythia) parity tests.

Usage:
    python3 test/fixtures/record_neox_parity.py [VARIANT]

VARIANT is one of: 70m, 410m, 1b, 6.9b (default: 70m).

Requirements:
    pip install transformers torch
    # Pythia and the original GPT-NeoX weights are open; no HF token needed.

Resource cost (fp32 weights + activations):
    70m:  ~0.3 GB download, ~0.4 GB RAM
    410m: ~1.6 GB download, ~1.8 GB RAM
    1b:   ~4 GB download,   ~4.5 GB RAM
    6.9b: ~28 GB download,  ~30 GB RAM   (server-class)

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA / FlashAttention so the
  accumulation order matches Allspark's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.
- add_special_tokens=False: Allspark's tokenizer doesn't auto-prepend
  any specials, so the recorder must not either.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANTS = {
    "70m":  ("EleutherAI/pythia-70m",   "pythia_70m_parity.json"),
    "410m": ("EleutherAI/pythia-410m",  "pythia_410m_parity.json"),
    "1b":   ("EleutherAI/pythia-1b",    "pythia_1b_parity.json"),
    "6.9b": ("EleutherAI/pythia-6.9b",  "pythia_6_9b_parity.json"),
}

PROMPT = "The capital of France is"
TOP_K = 50
TOLERANCE = 1e-3
FIXTURES_DIR = os.path.dirname(__file__)


def main(variant):
    if variant not in VARIANTS:
        print(
            f"Unknown variant: {variant!r}. Pick one of: {list(VARIANTS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_id, filename = VARIANTS[variant]
    out_path = os.path.join(FIXTURES_DIR, filename)

    print(f"Loading tokenizer for {repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Loading model {repo_id} in fp32 with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    encoded = tokenizer.encode(PROMPT, add_special_tokens=False)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs (no BOS): {encoded}")

    input_ids = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0, -1, :].float().cpu().numpy()

    top = sorted(range(len(logits)), key=lambda i: -logits[i])[:TOP_K]
    fixture = {
        "repo_id": repo_id,
        "prompt": PROMPT,
        "input_token_ids": encoded,
        "argmax_token_id": int(top[0]),
        "top_indices": [int(i) for i in top],
        "top_logits": [float(logits[i]) for i in top],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture to {out_path}")
    print(f"Argmax token id: {top[0]}  ({tokenizer.decode([top[0]])!r})")
    print(f"Top-{TOP_K} logit range: [{logits[top[-1]]:.4f}, {logits[top[0]]:.4f}]")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "70m"
    main(variant)
