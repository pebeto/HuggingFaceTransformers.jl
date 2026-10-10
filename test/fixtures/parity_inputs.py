"""Deterministic model inputs shared by the parity recorders.

Fixtures used to store a seeded `torch.randn` tensor, because Julia cannot
reproduce torch's random stream. Those tensors are the bulk of a fixture (a
3x518x518 image is 16 MB of JSON), and they belong to the input rather than the
answer. These closed-form patterns are mirrored exactly in
test/parity_inputs.jl, so a fixture only has to store the model's outputs.

Values are computed in float64 and cast to float32, matching what both sides
feed the model. The patterns vary along every axis so attention and position
embeddings see structure, which a constant input would not exercise.
"""
import numpy as np


def pixel_pattern(channels, height, width):
    """A (1, C, H, W) image-like tensor in PyTorch layout, roughly in [-1.5, 1.5]."""
    c, h, w = np.meshgrid(
        np.arange(channels), np.arange(height), np.arange(width), indexing="ij"
    )
    v = np.sin(0.31 * h + 0.17 * w + 1.3 * c) + 0.5 * np.cos(0.13 * h - 0.29 * w + 0.7 * c)
    return v.astype(np.float32)[None]


def feature_pattern(mel_bins, frames):
    """A (1, mel, frames) log-mel-like tensor in PyTorch layout, in [-1, 1]."""
    m, t = np.meshgrid(np.arange(mel_bins), np.arange(frames), indexing="ij")
    v = 0.6 * np.sin(0.19 * m + 0.031 * t) + 0.4 * np.cos(0.07 * m - 0.013 * t)
    return v.astype(np.float32)[None]


def image_pattern(height, width):
    """An (H, W, 3) uint8 image, the channels-last layout decoded images come in.

    Integer arithmetic only, so both languages agree on every byte. The quadratic
    phase runs smooth near the origin and wraps from 255 to 0 increasingly often
    further out, so a resize sees both gradients and hard edges (which make the
    bicubic kernel overshoot and clip).
    """
    h, w, c = np.meshgrid(np.arange(height), np.arange(width), np.arange(3), indexing="ij")
    return ((3 * h * h + 5 * w * w + 7 * h * w + 71 * c + 11) % 256).astype(np.uint8)
