#!/usr/bin/env python3
"""Record reference embeddings for Allspark.jl's nomic-embed parity test.

Usage:
    python3 test/fixtures/record_nomic_parity.py [VARIANT]

VARIANT is one of: v1.5 (default: v1.5).

Requirements:
    pip install transformers torch einops
    # NomicBERT is a custom architecture, so trust_remote_code is required.

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation stays default; nomic-bert ships its own attention. We
  assert parity in fp32.
- nomic-embed expects a task prefix; we use "search_document:".

The fixture records the special-token-bearing input IDs (Allspark feeds these
verbatim), and the mean-pooled, L2-normalized reference vector.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

VARIANTS = {
    "v1.5": ("nomic-ai/nomic-embed-text-v1.5", "nomic_embed_v1_5_parity.json"),
}
PROMPT = "search_document: how do rotary position embeddings work?"
TOLERANCE = 1e-3
FIXTURES_DIR = os.path.dirname(__file__)


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def main(variant):
    if variant not in VARIANTS:
        print(f"Unknown variant: {variant!r}. Pick one of: {list(VARIANTS)}",
              file=sys.stderr)
        sys.exit(1)

    repo_id, filename = VARIANTS[variant]
    out_path = os.path.join(FIXTURES_DIR, filename)

    print(f"Loading tokenizer for {repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Loading model {repo_id} in fp32 (trust_remote_code)...")
    model = AutoModel.from_pretrained(
        repo_id, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    encoded = tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs: {encoded}")

    input_ids = torch.tensor([encoded], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = mean_pool(out.last_hidden_state, attention_mask)
    emb = F.normalize(pooled, p=2, dim=1)[0].float().cpu().numpy()

    fixture = {
        "repo_id": repo_id,
        "prompt": PROMPT,
        "pooling": "mean",
        "normalize": True,
        "input_token_ids": encoded,
        "embedding": [float(x) for x in emb],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture to {out_path}")
    print(f"Embedding dim: {len(emb)}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1.5"
    main(variant)
