#!/usr/bin/env python3
"""Record reference log-mel features for HuggingFaceTransformers.jl's Whisper
feature-extractor parity test.

Usage:
    python3 test/fixtures/record_whisper_features_parity.py

Requirements:
    pip install transformers torch
    # With torch installed, HF computes features through `torch.stft`, which is
    # the path most users get, so that is the reference recorded here.

The input is a 0.5 s signal generated from a closed-form expression that the
Julia test reproduces exactly, so the audio itself is never stored. Features are
padded to 30 s (3000 frames), but everything after the audio ends collapses to a
single constant (silence floors at `max - 8`). The fixture therefore stores the
leading frames that carry signal plus that one plateau value, which describes the
full output losslessly in a few tens of kilobytes.

That size is deliberate. Fixtures holding full input tensors run to megabytes and
belong outside version control; this one is small enough to commit.
"""
import json
import os

import numpy as np
from transformers import WhisperFeatureExtractor

SAMPLING_RATE = 16_000
N_SAMPLES = 8_000                      # 0.5 s
HEAD_FRAMES = 60                       # 50 frames of audio plus the boundary
DECIMALS = 6
FIXTURES_DIR = os.path.dirname(__file__)
OUT = os.path.join(FIXTURES_DIR, "whisper_features_parity.json")


def signal():
    # Inharmonic partials plus a chirp, so every mel band sees energy. Mirrored
    # verbatim by `_test_signal` in test/parity_whisper_features.jl.
    t = np.arange(N_SAMPLES) / SAMPLING_RATE
    x = (0.4 * np.sin(2 * np.pi * 440 * t)
         + 0.25 * np.sin(2 * np.pi * 1250 * t)
         + 0.15 * np.sin(2 * np.pi * 3700 * t)
         + 0.2 * np.sin(2 * np.pi * (200 + 900 * t) * t))
    return x.astype(np.float32)


def main():
    fe = WhisperFeatureExtractor()
    features = fe(signal(), sampling_rate=SAMPLING_RATE, return_tensors="np").input_features[0]

    plateau = features[:, -1]
    assert np.allclose(features[:, HEAD_FRAMES:], plateau[:, None]), \
        "frames past the head are not constant; raise HEAD_FRAMES"
    assert np.allclose(plateau, plateau[0]), "the silent plateau is not a single value"

    fixture = {
        "feature_extractor": "WhisperFeatureExtractor",
        "sampling_rate": SAMPLING_RATE,
        "n_samples": N_SAMPLES,
        "shape": list(features.shape),
        "head_frames": HEAD_FRAMES,
        # Row-major (mel bin, frame), so the Julia side reads it as rows.
        # Round in float64: rounding a float32 array keeps float32, which then
        # serializes with ~16 digits and doubles the file for no information.
        "head": np.round(features[:, :HEAD_FRAMES].astype(np.float64), DECIMALS).tolist(),
        "plateau": round(float(plateau[0]), DECIMALS),
        "tolerance": 1e-4,
    }
    with open(OUT, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
    print(f"Wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
