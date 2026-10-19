#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's gpt-oss parity test.

Usage:
    python3 test/fixtures/record_gpt_oss_parity.py

Requirements:
    pip install transformers torch accelerate
    # ~13 GB download (openai/gpt-oss-20b), ~12 GB RAM

The full model needs ~42 GB once HF dequantizes its MXFP4 experts on a CPU, so
the reference is a two-layer cut of the real checkpoint: one sliding-window layer
and one full-attention layer, with the real embeddings, YaRN RoPE, sinks, router
and MXFP4 experts. That exercises every component the 24-layer model has. The
prompt runs past the 128-token window so the sliding layer actually drops keys,
and logits are recorded on both sides of that boundary.

It also records a greedy decode through HF's own KV cache: a harmony chat prompt
(rendered by the model's template, with the date pinned) followed by 150 new
tokens, so decoding crosses the window too. Each step keeps the chosen token and
the top two logits, letting the Julia side tell a real disagreement from a near
tie.

The reference runs in float32 throughout. On a CPU, HF decodes the MXFP4 experts
to bfloat16 and keeps them there (even when asked for float32), so its MoE runs
in bf16 and differs from a float32 one by ~0.4%. The decoded values are exact in
either type, so casting the experts up and taking the eager experts path gives
the float32 result, which is what HuggingFaceTransformers.jl computes.
"""
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

REPO = "openai/gpt-oss-20b"
N_LAYERS = 2
TOP_K = 20
N_NEW = 150
DATE = "2025-08-05"
CHAT = [{"role": "user", "content": "Name three rivers in Peru and say which is longest."}]
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIXTURES_DIR, "gpt_oss_20b_2layer_parity.json")

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
    config = AutoConfig.from_pretrained(REPO)
    config.num_hidden_layers = N_LAYERS
    config.layer_types = config.layer_types[:N_LAYERS]
    assert config.layer_types == ["sliding_attention", "full_attention"], config.layer_types

    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, config=config, dtype=torch.float32, attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True),
    ).eval()
    model.float()
    model.config._experts_implementation = "eager"
    experts = model.model.layers[0].mlp.experts
    assert experts.gate_up_proj.dtype == torch.float32, experts.gate_up_proj.dtype

    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    window = config.sliding_window
    assert len(ids) > window + 32, f"prompt has {len(ids)} tokens; needs to pass the {window}-token window"
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()

    prompt_text = tokenizer.apply_chat_template(
        CHAT, add_generation_prompt=True, tokenize=False, reasoning_effort="low",
        strftime_now=lambda fmt: DATE,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    assert len(prompt_ids) < window < len(prompt_ids) + N_NEW, len(prompt_ids)
    tokens, top2 = [], []
    with torch.no_grad():
        out = model(torch.tensor([prompt_ids]), use_cache=True)
        for _ in range(N_NEW):
            values, indices = torch.topk(out.logits[0, -1].float(), 2)
            tokens.append(int(indices[0]))
            top2.append([float(v) for v in values])
            out = model(torch.tensor([[tokens[-1]]]), past_key_values=out.past_key_values, use_cache=True)
    generation = {
        "messages": CHAT,
        "reasoning_effort": "low",
        "date": DATE,
        "prompt_ids": prompt_ids,
        "tokens": tokens,
        "top2": top2,
    }

    positions = [16, window - 1, window, window + 16, len(ids) - 1]
    records = []
    for p in positions:
        values, indices = torch.topk(logits[p], TOP_K)
        records.append({
            "position": p,
            "top_indices": indices.tolist(),
            "top_logits": [float(v) for v in values],
        })
    fixture = {
        "repo_id": REPO,
        "num_hidden_layers": N_LAYERS,
        "prompt": PROMPT,
        "input_token_ids": ids,
        "positions": records,
        "generation": generation,
        "tolerance": 1e-4,
    }
    with open(OUT, "w") as f:
        json.dump(fixture, f, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(ids)} tokens; wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")
    print(f"generation: {len(prompt_ids)} prompt tokens + {N_NEW} new")
    for r in records:
        print(r["position"], r["top_indices"][:3], [round(v, 4) for v in r["top_logits"][:3]])


if __name__ == "__main__":
    main()
