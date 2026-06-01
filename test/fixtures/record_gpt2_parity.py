#!/usr/bin/env python3
"""Record reference logits for Allspark.jl's GPT-2 parity tests.

Usage:
    python3 test/fixtures/record_gpt2_parity.py [VARIANT]

VARIANT is one of: 124M, 355M, 774M, 1558M (default: 124M).

Requirements:
    pip install transformers torch
    # All GPT-2 sizes are open weights; no HF token needed.

Resource cost (fp32 weights + activations):
    124M:  ~0.5 GB download, ~0.6 GB RAM
    355M:  ~1.5 GB download, ~1.7 GB RAM
    774M:  ~3 GB download,   ~3.5 GB RAM
    1558M: ~6 GB download,   ~7 GB RAM

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
    "124M":  ("gpt2",         "gpt2_124m_parity.json"),
    "355M":  ("gpt2-medium",  "gpt2_355m_parity.json"),
    "774M":  ("gpt2-large",   "gpt2_774m_parity.json"),
    "1558M": ("gpt2-xl",      "gpt2_1558m_parity.json"),
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
    variant = sys.argv[1] if len(sys.argv) > 1 else "124M"
    main(variant)
