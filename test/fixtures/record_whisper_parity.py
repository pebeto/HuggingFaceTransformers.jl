#!/usr/bin/env python3
"""Record reference logits for Allspark.jl's Whisper parity test.

Usage:
    python3 test/fixtures/record_whisper_parity.py [VARIANT]

VARIANT is one of: base (default: base).

Requirements:
    pip install transformers torch

Why these choices (matches the other record_*_parity.py scripts):
- attn_implementation="eager": disables SDPA so accumulation order matches
  Allspark's naive softmax attention.
- torch_dtype=torch.float32: parity is asserted in fp32.

The fixture stores a seeded random `input_features` tensor (audio → log-mel is
out of scope) plus a fixed `decoder_input_ids` prompt, and the last-position
logits. `input_features` is a C-order flat list plus its `(N, mel, frames)`
shape; at native size it is ~0.24M floats, so the fixture is a few MB and is
generated locally, not committed.
"""
import json
import os
import sys

import torch
from transformers import WhisperForConditionalGeneration

VARIANTS = {
    "base": ("openai/whisper-base", "whisper_base_parity.json"),
}
# SOT prompt: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
DECODER_INPUT_IDS = [50258, 50259, 50359, 50363]
TOP_K = 50
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
    model = WhisperForConditionalGeneration.from_pretrained(
        repo_id, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    n_mels = model.config.num_mel_bins
    frames = 2 * model.config.max_source_positions   # conv2 halves the length
    gen = torch.Generator().manual_seed(0)
    input_features = torch.randn(1, n_mels, frames, generator=gen)
    decoder_input_ids = torch.tensor([DECODER_INPUT_IDS], dtype=torch.long)

    with torch.no_grad():
        out = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
    logits = out.logits[0, -1, :].float().cpu()      # last decoder position

    top = torch.topk(logits, TOP_K)
    fixture = {
        "repo_id": repo_id,
        "feature_shape": list(input_features.shape),
        "input_features": input_features.flatten().tolist(),
        "decoder_input_ids": DECODER_INPUT_IDS,
        "argmax_token_id": int(logits.argmax()),
        "top_indices": [int(i) for i in top.indices.tolist()],
        "top_logits": [float(v) for v in top.values.tolist()],
        "tolerance": TOLERANCE,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(f"Wrote fixture to {out_path}")
    print(f"Argmax token id: {int(logits.argmax())}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    main(variant)
