#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's Phi-3 parity tests.

Usage:
    python3 test/fixtures/record_phi3_parity.py [VARIANT]

VARIANT is one of: mini-4k, medium-4k (default: mini-4k).

Only the 4k context variants are recorded — the 128k variants use the
`longrope` scaling that HuggingFaceTransformers doesn't implement yet. Phi-3.5 also
needs partial RoPE work.

Requirements:
    pip install transformers torch
    huggingface-cli login         # or set HF_TOKEN

Resource cost (fp32 weights + activations):
    mini-4k   (3.8B): ~16 GB download, ~18 GB RAM
    medium-4k (14B):  ~56 GB download, ~60 GB RAM   (server-class)

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA / FlashAttention so the
  accumulation order matches HuggingFaceTransformers's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.
- add_special_tokens=False: HuggingFaceTransformers's tokenizer doesn't auto-prepend
  any specials, so the recorder must not either.
- trust_remote_code=True: Phi-3 still uses custom modeling code on Hub.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANTS = {
    "mini-4k":   ("microsoft/Phi-3-mini-4k-instruct",   "phi_3_mini_4k_instruct_parity.json"),
    "medium-4k": ("microsoft/Phi-3-medium-4k-instruct", "phi_3_medium_4k_instruct_parity.json"),
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
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    print(f"Loading model {repo_id} in fp32 with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
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
    variant = sys.argv[1] if len(sys.argv) > 1 else "mini-4k"
    main(variant)
