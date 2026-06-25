#!/usr/bin/env python3
"""Record reference sentence embeddings for Allspark.jl's embedding parity tests.

Usage:
    python3 test/fixtures/record_embedding_parity.py [VARIANT]

VARIANT is one of: bge-small, e5-small (default: bge-small).

Requirements:
    pip install transformers torch
    # Both checkpoints are open; no HF token needed.

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA / FlashAttention so the
  accumulation order matches Allspark's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.

The fixture records the special-token-bearing input IDs (Allspark feeds these
verbatim, so tokenizer special-token handling isn't on the critical path), the
pooling strategy, the detected state-dict prefix, and the unit-norm reference
vector. BGE uses CLS pooling; E5 uses mean pooling and a "query:" prefix.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

VARIANTS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "cls", "bge_small_en_parity.json",
                  "The quick brown fox jumps over the lazy dog."),
    "e5-small":  ("intfloat/e5-small-v2", "mean", "e5_small_v2_parity.json",
                  "query: how do transformer embedding models work?"),
}
TOLERANCE = 1e-3
FIXTURES_DIR = os.path.dirname(__file__)


def pool(last_hidden, attention_mask, kind):
    if kind == "cls":
        return last_hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def main(variant):
    if variant not in VARIANTS:
        print(f"Unknown variant: {variant!r}. Pick one of: {list(VARIANTS)}",
              file=sys.stderr)
        sys.exit(1)

    repo_id, pooling, filename, prompt = VARIANTS[variant]
    out_path = os.path.join(FIXTURES_DIR, filename)

    print(f"Loading tokenizer for {repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Loading model {repo_id} in fp32 with eager attention...")
    model = AutoModel.from_pretrained(
        repo_id, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Prompt: {prompt!r}")
    print(f"Token IDs: {encoded}")

    input_ids = torch.tensor([encoded], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = pool(out.last_hidden_state, attention_mask, pooling)
    emb = F.normalize(pooled, p=2, dim=1)[0].float().cpu().numpy()

    # Detect whether the checkpoint stores the trunk under a `bert.`/`roberta.`
    # namespace or flat (BGE/E5 are usually flat).
    keys = list(model.state_dict().keys())
    if any(k.startswith("embeddings.") for k in keys):
        hf_prefix = ""
    elif any(k.startswith("roberta.") for k in keys):
        hf_prefix = "roberta"
    else:
        hf_prefix = "bert"
    offset = 2 if hf_prefix == "roberta" else 0

    fixture = {
        "repo_id": repo_id,
        "prompt": prompt,
        "pooling": pooling,
        "normalize": True,
        "hf_prefix": hf_prefix,
        "position_embedding_offset": offset,
        "input_token_ids": encoded,
        "embedding": [float(x) for x in emb],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture to {out_path}")
    print(f"Pooling: {pooling}, prefix: {hf_prefix!r}, dim: {len(emb)}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "bge-small"
    main(variant)
