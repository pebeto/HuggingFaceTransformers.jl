#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's BERT/RoBERTa parity tests.

Records RoBERTa only. WordPiece tokenization is supported, so BERT proper is
no longer blocked on the tokenizer; it just has no reference recorded here.

Usage:
    python3 test/fixtures/record_bert_parity.py [VARIANT]

VARIANT is one of: base, large (default: base).

Requirements:
    pip install transformers torch
    # RoBERTa weights are open; no HF token needed.

Resource cost (fp32 weights + activations):
    base:  ~0.5 GB download, ~0.6 GB RAM  (125M params)
    large: ~1.5 GB download, ~1.7 GB RAM  (355M)

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA / FlashAttention so the
  accumulation order matches HuggingFaceTransformers's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.

The prompt contains an explicit `<mask>` token whose logits we compare;
that's how MaskedLM models are evaluated. The fixture records both the
prompt's tokenization and the 0-indexed position of the mask within it.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

VARIANTS = {
    "base":  ("FacebookAI/roberta-base",   "roberta_base_parity.json"),
    "large": ("FacebookAI/roberta-large",  "roberta_large_parity.json"),
}

# The mask token differs by tokenizer (RoBERTa: `<mask>`). The prompt
# is built so the mask sits at a predictable spot, but the recorder
# locates it dynamically.
PROMPT = "The capital of France is <mask>."
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
    model = AutoModelForMaskedLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    # HF tokenizes with special tokens (BOS=<s>, EOS=</s>) for RoBERTa
    # masked-LM evaluation. We record what HF actually feeds the model.
    encoded = tokenizer.encode(PROMPT, add_special_tokens=True)
    mask_index = encoded.index(tokenizer.mask_token_id)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs: {encoded}")
    print(f"Mask position (0-indexed): {mask_index}")

    input_ids = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0, mask_index, :].float().cpu().numpy()

    top = sorted(range(len(logits)), key=lambda i: -logits[i])[:TOP_K]
    fixture = {
        "repo_id": repo_id,
        "prompt": PROMPT,
        "input_token_ids": encoded,
        "mask_index": int(mask_index),
        "argmax_token_id": int(top[0]),
        "top_indices": [int(i) for i in top],
        "top_logits": [float(logits[i]) for i in top],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture to {out_path}")
    print(f"Argmax token id: {top[0]}  ({tokenizer.decode([top[0]])!r})")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    main(variant)
