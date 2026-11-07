#!/usr/bin/env python3
"""Record reference logits for HuggingFaceTransformers.jl's quantized Qwen3.5 parity
test, from the first layers of a compressed-tensors NVFP4/FP8 checkpoint.

Usage:
    python3 test/fixtures/record_qwen3_5_quantized_parity.py [SNAPSHOT_DIR]

Requirements:
    pip install transformers torch safetensors compressed-tensors
    # ~22 GB download (unsloth/Qwen3.8-27B-NVFP4), ~14 GB RAM

The checkpoint mixes formats: NVFP4 MLPs (4-bit E2M1 in groups of 16 along the
input, each group scaled by an FP8 E4M3 scale divided by a per-tensor F32 global
scale), FP8 E4M3 everywhere else with a per-output-channel scale, and bf16
embeddings. HuggingFaceTransformers.jl decodes these weights exactly and computes
in float32 with full-precision activations.

HF's own CPU path cannot be that reference: it decodes NVFP4 to bf16, which rounds
every weight, and fake-quantizes activations (FP8 per token, NVFP4 per group of
16), running the checkpoint as W4A4/W8A8 the way FP4 hardware would. So this
script decodes the weights exactly itself (after checking its decoder against
compressed-tensors', which must agree to bf16 rounding), loads them into HF's own
Qwen3.5 text model, and runs that in float32. That isolates the architecture and
the decoding from both of HF's approximations.

The cut is the first four layers: three Gated DeltaNet layers (48 value heads
sharing 16 key heads) and one gated full-attention layer, with NVFP4 MLPs and
FP8 projections, plus the real embeddings, final norm and FP8 LM head.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize

REPO = "unsloth/Qwen3.8-27B-NVFP4"
N_LAYERS = 4
TOP_K = 20
N_NEW = 40
PREFIX = "model.language_model."
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIXTURES_DIR, "qwen3_5_27b_nvfp4_4layer_parity.json")
CHAT = [{"role": "user", "content": "Name three rivers in Peru and say which is longest."}]
PROMPT = (
    "The lighthouse keeper wrote in his log every evening, even when nothing happened. "
    "On the first of March the wind turned east and the fishing boats stayed in the harbour. "
    "On the second, a crate of oranges washed ashore, and the children of the village carried "
    "them up the hill in their sweaters."
)
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def nvfp4_exact(packed, scale, global_scale):
    values = torch.stack([FP4[(packed & 0x0F).long()], FP4[(packed >> 4).long()]], dim=-1)
    values = values.reshape(packed.shape[0], -1)
    return values * (scale.float() / global_scale.float()).repeat_interleave(16, dim=1)


def fp8_exact(weight, scale):
    return weight.float() * scale.float()


class Checkpoint:
    def __init__(self, path):
        index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.path, self.index, self.files = path, index, {}

    def has(self, key):
        return key in self.index

    def get(self, key):
        name = self.index[key]
        if name not in self.files:
            self.files[name] = safe_open(os.path.join(self.path, name), "pt")
        return self.files[name].get_tensor(key)

    def linear(self, key):
        """Exact float32 weight of a linear layer, whatever its format."""
        if self.has(key + ".weight_packed"):
            return nvfp4_exact(self.get(key + ".weight_packed"), self.get(key + ".weight_scale"),
                               self.get(key + ".weight_global_scale"))
        weight = self.get(key + ".weight")
        if weight.dtype == torch.float8_e4m3fn:
            return fp8_exact(weight, self.get(key + ".weight_scale"))
        return weight.float()


def check_decoders(ckpt):
    """Our exact decoding must agree with compressed-tensors' to its bf16 rounding."""
    key = PREFIX + "layers.0.mlp.gate_proj"
    packed = ckpt.get(key + ".weight_packed")
    scale, global_scale = ckpt.get(key + ".weight_scale"), ckpt.get(key + ".weight_global_scale")
    args = QuantizationArgs(num_bits=4, type="float", strategy="tensor_group", group_size=16,
                            scale_dtype=torch.float8_e4m3fn)
    library = dequantize(unpack_fp4_from_uint8(packed, packed.shape[0], packed.shape[1] * 2),
                         scale=scale.to(torch.bfloat16), args=args, global_scale=global_scale,
                         dtype=torch.bfloat16).float()
    exact = nvfp4_exact(packed, scale, global_scale)
    rel = ((library - exact).abs() / exact.abs().clamp_min(1e-30)).max().item()
    assert rel <= 2 ** -8, rel
    print(f"NVFP4 decoder agrees with compressed-tensors to {rel:.2e} (bf16 rounding)")


def main(path):
    ckpt = Checkpoint(path)
    check_decoders(ckpt)

    config = AutoConfig.from_pretrained(path).text_config
    config.num_hidden_layers = N_LAYERS
    config.layer_types = config.layer_types[:N_LAYERS]
    vocab = config.vocab_size
    config.vocab_size = 16       # embeddings are looked up here, from the checkpoint
    model = Qwen3_5TextModel(config).float().eval()
    state = {}
    for name, param in model.state_dict().items():
        if name.startswith("embed_tokens"):
            continue
        key = PREFIX + name
        if name.endswith(".weight") and not ckpt.has(key) and ckpt.has(key[:-7] + ".weight_packed"):
            state[name] = ckpt.linear(key[:-7])
        elif name.endswith(".weight") and ckpt.has(key) and ckpt.get(key).dtype == torch.float8_e4m3fn:
            state[name] = ckpt.linear(key[:-7])
        else:
            state[name] = ckpt.get(key).float()
        assert state[name].shape == param.shape, (name, state[name].shape, param.shape)
    model.load_state_dict(state, strict=False)
    del state
    embed = ckpt.get(PREFIX + "embed_tokens.weight")            # bf16, exact in float32
    lm_head = ckpt.linear("lm_head")                             # (vocab, hidden), float32
    assert lm_head.shape[0] == vocab

    def logits_for(input_ids, past=None):
        out = model(inputs_embeds=embed[torch.tensor(input_ids)].float()[None], past_key_values=past,
                    use_cache=True)
        return out.last_hidden_state[0] @ lm_head.T, out.past_key_values

    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    with torch.no_grad():
        logits, _ = logits_for(ids)
    records = []
    for p in [0, len(ids) // 2, len(ids) - 1]:
        values, indices = torch.topk(logits[p], TOP_K)
        records.append({"position": p, "top_indices": indices.tolist(),
                        "top_logits": [float(v) for v in values]})

    prompt_text = tokenizer.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    tokens, top2 = [], []
    with torch.no_grad():
        logits, past = logits_for(prompt_ids)
        for _ in range(N_NEW):
            values, indices = torch.topk(logits[-1], 2)
            tokens.append(int(indices[0]))
            top2.append([float(v) for v in values])
            logits, past = logits_for([tokens[-1]], past)

    fixture = {
        "repo_id": REPO,
        "num_hidden_layers": N_LAYERS,
        "prompt": PROMPT,
        "input_token_ids": ids,
        "positions": records,
        "generation": {"messages": CHAT, "prompt_ids": prompt_ids, "tokens": tokens, "top2": top2},
        "tolerance": 1e-4,
    }
    with open(OUT, "w") as f:
        json.dump(fixture, f, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(ids)} tokens; chat prompt {len(prompt_ids)} + {N_NEW}; wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")
    for r in records:
        print(r["position"], r["top_indices"][:3], [round(v, 4) for v in r["top_logits"][:3]])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        snapshot = sys.argv[1]
    else:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(REPO)
    with torch.no_grad():
        main(snapshot)
