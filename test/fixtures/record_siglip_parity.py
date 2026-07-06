#!/usr/bin/env python3
"""Record reference logits for Allspark.jl's SigLIP parity test.

Usage:
    python3 test/fixtures/record_siglip_parity.py [VARIANT]

VARIANT is one of: base (default: base).

Requirements:
    pip install transformers torch sentencepiece

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA so accumulation order matches
  Allspark's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.

The fixture stores a seeded random `pixel_values` tensor (image decode/resize is
out of scope) plus the tokenizer's padded (max_length=64) `input_ids`, so the
Julia side feeds both verbatim and verifies the full image + text → logits path.
"""
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

VARIANTS = {
    "base": ("google/siglip-base-patch16-224", "siglip_base_patch16_224_parity.json"),
}
TEXTS = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
TOLERANCE = 1e-2
FIXTURES_DIR = os.path.dirname(__file__)


def main(variant):
    if variant not in VARIANTS:
        print(f"Unknown variant: {variant!r}. Pick one of: {list(VARIANTS)}",
              file=sys.stderr)
        sys.exit(1)

    repo_id, filename = VARIANTS[variant]
    out_path = os.path.join(FIXTURES_DIR, filename)

    print(f"Loading model {repo_id} in fp32 with eager attention...")
    model = AutoModel.from_pretrained(
        repo_id, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    size = model.config.vision_config.image_size
    gen = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(1, 3, size, size, generator=gen)

    max_len = model.config.text_config.max_position_embeddings
    text_inputs = tokenizer(TEXTS, padding="max_length", max_length=max_len,
                            return_tensors="pt")

    with torch.no_grad():
        out = model(pixel_values=pixel_values, input_ids=text_inputs.input_ids)
    logits_per_image = out.logits_per_image[0].float().cpu()  # (n_texts,)

    fixture = {
        "repo_id": repo_id,
        "texts": TEXTS,
        "pixel_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "input_ids": text_inputs.input_ids.tolist(),   # (n_texts, max_len)
        "logits_per_image": [float(x) for x in logits_per_image.tolist()],
        "argmax_text": int(logits_per_image.argmax()),
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(f"Wrote fixture to {out_path}")
    print(f"Argmax text: {TEXTS[int(logits_per_image.argmax())]!r}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    main(variant)
