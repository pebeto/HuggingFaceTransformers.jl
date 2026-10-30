#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's Qwen3.5 parity test.

Usage:
    python3 test/fixtures/record_qwen3_5_parity.py

Requirements:
    pip install transformers torch
    # ~1.7 GB download (Qwen/Qwen3.5-0.8B), ~5 GB RAM

Qwen3.5 interleaves Gated DeltaNet linear attention with gated full attention.
The prompt is long enough (well past HF's 64-token chunks) that HF's chunked
delta rule spans several chunks, and logits are recorded across it. A second
trace renders a chat through the model's own template and decodes 100 greedy
tokens through HF's cache, which carries the delta nets' convolution windows and
recurrent states from step to step. Each step keeps the chosen token and the top
two logits, so the Julia side can tell a real disagreement from a near tie.

Everything runs in float32 with eager attention and HF's pure-PyTorch delta rule.
"""
import json
import os

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

REPO = "Qwen/Qwen3.5-0.8B"
TOP_K = 20
N_NEW = 100
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIXTURES_DIR, "qwen3_5_0_8b_parity.json")
CHAT = [{"role": "user", "content": "Name three rivers in Peru and say which is longest."}]

PROMPT = (
    "The lighthouse keeper wrote in his log every evening, even when nothing happened. "
    "On the first of March the wind turned east and the fishing boats stayed in the harbour. "
    "On the second, a crate of oranges washed ashore, and the children of the village carried "
    "them up the hill in their sweaters. On the third, the lamp's clockwork slowed, so he oiled "
    "the gears and counted the seconds between flashes until they were even again. On the fourth, "
    "a letter arrived from his sister in Valparaíso asking whether the sea was still the same "
    "colour. He answered that it changed every hour, grey at dawn, green at noon, and almost black "
    "before a storm, and that he had stopped trying to name it. On the fifth, nothing happened, and "
    "he wrote that down too, because a quiet day was also worth remembering. By the end of the month "
    "the log held thirty-one entries, and he read them aloud to the gulls on the gallery rail."
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        REPO, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    assert len(ids) > 128, len(ids)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    records = []
    for p in [0, 15, 63, 64, 100, len(ids) - 1]:
        values, indices = torch.topk(logits[p], TOP_K)
        records.append({"position": p, "top_indices": indices.tolist(),
                        "top_logits": [float(v) for v in values]})

    prompt_text = tokenizer.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    tokens, top2 = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        for _ in range(N_NEW):
            values, indices = torch.topk(out.logits[0, -1].float(), 2)
            tokens.append(int(indices[0]))
            top2.append([float(v) for v in values])
            out = model(input_ids=torch.tensor([[tokens[-1]]]), past_key_values=out.past_key_values,
                        use_cache=True)

    fixture = {
        "repo_id": REPO,
        "prompt": PROMPT,
        "input_token_ids": ids,
        "positions": records,
        "generation": {"messages": CHAT, "prompt_ids": prompt_ids, "tokens": tokens, "top2": top2},
        "tolerance": 1e-4,
    }
    with open(OUT, "w") as f:
        json.dump(fixture, f, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(ids)} tokens; chat prompt {len(prompt_ids)} + {N_NEW}; wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")
    print("decoded:", repr(tokenizer.decode(tokens)))
    for r in records:
        print(r["position"], r["top_indices"][:3], [round(v, 4) for v in r["top_logits"][:3]])


if __name__ == "__main__":
    main()
