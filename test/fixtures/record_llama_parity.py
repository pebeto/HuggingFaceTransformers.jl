#!/usr/bin/env python3
"""Record reference logits for Allspark.jl's Llama-3.2-1B parity test.

Run once in a Python environment that has the model cached. Writes
test/fixtures/llama_3_2_1b_parity.json, which the Julia parity test
reads as ground truth.

Requirements:
    pip install transformers torch
    huggingface-cli login         # or set HF_TOKEN

Why these choices:
- attn_implementation="eager": disables SDPA / FlashAttention. Allspark
  uses naive softmax-attention; eager makes PyTorch do the same so
  floating-point accumulation order matches.
- torch_dtype=torch.float32: parity is asserted in fp32.
- add_special_tokens=False: Allspark's tokenizer doesn't auto-prepend
  BOS, so the recorder must not either.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
TOP_K = 50
TOLERANCE = 1e-3
OUT_PATH = os.path.join(os.path.dirname(__file__), "llama_3_2_1b_parity.json")


def main():
    print(f"Loading tokenizer for {REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    print(f"Loading model {REPO_ID} in fp32 with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    encoded = tokenizer.encode(PROMPT, add_special_tokens=False)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs (no BOS): {encoded}")

    input_ids = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0, -1, :].float().cpu().numpy()

    top = sorted(range(len(logits)), key=lambda i: -logits[i])[:TOP_K]
    fixture = {
        "repo_id": REPO_ID,
        "prompt": PROMPT,
        "input_token_ids": encoded,
        "argmax_token_id": int(top[0]),
        "top_indices": [int(i) for i in top],
        "top_logits": [float(logits[i]) for i in top],
        "tolerance": TOLERANCE,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture to {OUT_PATH}")
    print(f"Argmax token id: {top[0]}  ({tokenizer.decode([top[0]])!r})")
    print(f"Top-{TOP_K} logit range: [{logits[top[-1]]:.4f}, {logits[top[0]]:.4f}]")


if __name__ == "__main__":
    main()
