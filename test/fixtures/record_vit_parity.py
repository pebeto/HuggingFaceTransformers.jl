#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's ViT parity test.

Usage:
    python3 test/fixtures/record_vit_parity.py [VARIANT]

VARIANT is one of: base (default: base).

Requirements:
    pip install transformers torch

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA so accumulation order matches
  HuggingFaceTransformers's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.

The input is `pixel_pattern` from parity_inputs.py, a closed-form tensor the
Julia side regenerates bit for bit, so the fixture stores only its shape and the
model's outputs. Storing the tensor itself made this file several megabytes.
"""
import json
import os
import sys

# The shared input patterns live beside this script. Added explicitly so the
# recorder still finds them when run in isolated mode (`python -I`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_inputs import pixel_pattern  # noqa: E402

import torch
from transformers import AutoModelForImageClassification

VARIANTS = {
    "base": ("google/vit-base-patch16-224", "vit_base_patch16_224_parity.json"),
}
TOP_K = 50
TOLERANCE = 1e-3
FIXTURES_DIR = os.path.dirname(__file__)


def main(variant):
    if variant not in VARIANTS:
        print(f"Unknown variant: {variant!r}. Pick one of: {list(VARIANTS)}",
              file=sys.stderr)
        sys.exit(1)

    repo_id, filename = VARIANTS[variant]
    out_path = os.path.join(FIXTURES_DIR, filename)

    print(f"Loading model {repo_id} in fp32 with eager attention...")
    model = AutoModelForImageClassification.from_pretrained(
        repo_id, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    size = model.config.image_size
    pixel_values = torch.from_numpy(pixel_pattern(3, size, size))
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits[0].float().cpu()

    top = torch.topk(logits, TOP_K)
    fixture = {
        "repo_id": repo_id,
        "pixel_shape": list(pixel_values.shape),
        "input": "pixel_pattern",
        "argmax_label": int(logits.argmax()),
        "top_indices": [int(i) for i in top.indices.tolist()],
        "top_logits": [float(v) for v in top.values.tolist()],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(f"Wrote fixture to {out_path}")
    print(f"Argmax label: {int(logits.argmax())} ({model.config.id2label[int(logits.argmax())]})")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    main(variant)
