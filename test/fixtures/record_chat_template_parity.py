#!/usr/bin/env python3
"""Record reference renderings for HuggingFaceTransformers.jl's chat template
parity test.

Usage:
    python3 test/fixtures/record_chat_template_parity.py

Requirements:
    pip install transformers jinja2

Only tokenizer files are downloaded. Each template's source is stored in the
fixture next to HF's `apply_chat_template` output for a set of conversations, so
the Julia test runs offline in the default suite. The conversations cover what
these templates branch on: a system prompt, multi-turn history, Qwen3's thinking
blocks (inline and as `reasoning_content`), tool schemas, tool calls with dict and
string arguments, and consecutive tool responses.
"""
import json
import os

from transformers import AutoTokenizer

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIXTURES_DIR, "chat_template_parity.json")

TEMPLATES = [
    ("qwen3", "Qwen/Qwen3-0.6B"),
    ("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct"),
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city, in °C. Say \"unknown\" if unsure.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Lima"},
                "precision": {"type": "number", "default": 0.5},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
            },
            "required": ["city"],
        },
    },
}

CASES = [
    ("system_and_user", {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Name a prime."},
        ],
        "add_generation_prompt": True,
    }),
    ("thinking_disabled", {
        "messages": [{"role": "user", "content": "Hi!\nTwo lines."}],
        "add_generation_prompt": True,
        "kwargs": {"enable_thinking": False},
    }),
    ("history_with_inline_thinking", {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>\nAdd them.\n</think>\n\n4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("reasoning_content_last_turn", {
        "messages": [
            {"role": "user", "content": "Pick a color."},
            {"role": "assistant", "reasoning_content": "\nAny will do.\n", "content": "Blue."},
        ],
        "add_generation_prompt": False,
    }),
    ("tools_and_calls", {
        "messages": [
            {"role": "system", "content": "Use tools when helpful."},
            {"role": "user", "content": "Weather in Lima and Cusco?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lima"}}},
                {"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Cusco\", \"days\": 2}"}},
            ]},
            {"role": "tool", "content": "{\"temp\": 19.5}"},
            {"role": "tool", "content": "{\"temp\": 11}"},
            {"role": "assistant", "content": "Lima 19.5 °C, Cusco 11 °C."},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": [WEATHER_TOOL],
        "add_generation_prompt": True,
    }),
    ("assistant_text_then_call", {
        "messages": [
            {"role": "user", "content": "Check Lima."},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lima", "precision": 1e-05}}},
            ]},
            {"role": "tool", "content": "sunny"},
        ],
        "tools": [WEATHER_TOOL],
        "add_generation_prompt": True,
    }),
]


def main():
    fixture = {"templates": []}
    for name, repo in TEMPLATES:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        template = tokenizer.chat_template
        cases = []
        for case_name, case in CASES:
            rendered = tokenizer.apply_chat_template(
                case["messages"],
                tools=case.get("tools"),
                add_generation_prompt=case["add_generation_prompt"],
                tokenize=False,
                **case.get("kwargs", {}),
            )
            cases.append({"name": case_name, **case, "expected": rendered})
        fixture["templates"].append({"name": name, "repo_id": repo, "template": template, "cases": cases})
        print(f"{name}: {len(cases)} cases from {repo}")
    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1, ensure_ascii=False)
    print(f"Wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
