#!/usr/bin/env python3
"""Record the reference CLS feature for Allspark.jl's DINOv2 parity test.

Usage:
    python3 test/fixtures/record_dinov2_parity.py [VARIANT]

VARIANT is one of: base (default: base).

Requirements:
    pip install transformers torch

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA so accumulation order matches
  Allspark's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.

The fixture stores a seeded random `pixel_values` tensor at the model's native
resolution (image decode/resize is out of scope) and HF's `pooler_output` (the
CLS token after the final LayerNorm). Note: at native resolution the pixel
tensor is large (dinov2-base is 518x518, ~0.8M floats), so the fixture is a few
MB; it is generated locally and not committed.
"""
import json
import os
import sys

import torch
from transformers import AutoModel

VARIANTS = {
    "base": ("facebook/dinov2-base", "dinov2_base_parity.json"),
}
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

    size = model.config.image_size
    gen = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(1, 3, size, size, generator=gen)
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    pooler = out.pooler_output[0].float().cpu()   # CLS after final LayerNorm

    fixture = {
        "repo_id": repo_id,
        "pixel_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "pooler_output": [float(x) for x in pooler.tolist()],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(f"Wrote fixture to {out_path}")
    print(f"CLS feature dim: {len(pooler)}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    main(variant)
