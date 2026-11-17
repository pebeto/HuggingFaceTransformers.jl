#!/usr/bin/env python3
"""Record quantized blocks for HuggingFaceTransformers.jl's GGUF decoding test.

Usage:
    python3 test/fixtures/record_gguf_blocks.py

Requirements:
    pip install gguf huggingface_hub

For every tensor type the Julia reader decodes, this takes the first two blocks
of the first tensor of that type in the GGUF files below, and stores their bytes
with the values gguf-py's reference `dequantize` gives for them. The test decodes
the same bytes and expects the same Float32 values exactly. The files are those
of record_gguf_parity.py plus SmolLM2's F16, Q4_0, Q5_K_M and Q6_K quantizations,
which together hold every type.
"""
import json
import os

import gguf
import numpy as np
from huggingface_hub import hf_hub_download

FILES = [
    ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-f16.gguf"),
    ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q4_0.gguf"),
    ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q3_K_M.gguf"),
    ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q5_K_M.gguf"),
    ("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q6_K.gguf"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q2_K.gguf"),
]
BLOCKS = 2
OUT = os.path.join(os.path.dirname(__file__), "gguf_quant_blocks.json")


def main():
    found = {}
    for repo, name in FILES:
        reader = gguf.GGUFReader(hf_hub_download(repo, name))
        for t in reader.tensors:
            kind = t.tensor_type
            if kind.name in found:
                continue
            _, block_bytes = gguf.GGML_QUANT_SIZES[kind]
            raw = np.asarray(t.data).view(np.uint8).ravel()[: BLOCKS * block_bytes]
            values = gguf.quants.dequantize(raw, kind).astype(np.float32).ravel()
            found[kind.name] = {
                "type": int(kind.value),
                "source": f"{repo}/{name}:{t.name}",
                "bytes": raw.tobytes().hex(),
                "values": [float(v) for v in values],
            }
    with open(OUT, "w") as f:
        json.dump(dict(sorted(found.items())), f, indent=1)
    print(f"wrote {sorted(found)} to {OUT}")


if __name__ == "__main__":
    main()
