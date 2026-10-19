#!/usr/bin/env python3
"""Record reference renderings for HuggingFaceTransformers.jl's chat template
parity test.

Usage:
    python3 test/fixtures/record_chat_template_parity.py [NAME ...]

With names (qwen3, qwen2.5, gpt-oss), only those templates are re-recorded and
the rest of the fixture is kept.

Requirements:
    pip install transformers jinja2

Only tokenizer files are downloaded. Each template's source is stored in the
fixture next to HF's `apply_chat_template` output for a set of conversations, so
the Julia test runs offline in the default suite. The conversations cover what
these templates branch on: a system prompt, multi-turn history, Qwen3's thinking
blocks (inline and as `reasoning_content`), tool schemas, tool calls with dict and
string arguments, and consecutive tool responses. gpt-oss's harmony template gets
its own set: developer instructions, model identity and reasoning effort, dropped
and kept analysis, tool schemas rendered as TypeScript (enums, defaults, unions,
nested objects and arrays), tool calls and results, and the built-in browser and
python tools. Its system message carries the current date, so those cases pin
`strftime_now` and the fixture records the date used.
"""
import json
import os
import sys

from transformers import AutoTokenizer

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIXTURES_DIR, "chat_template_parity.json")


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

QWEN_CASES = [
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


DATE = "2025-08-05"

HARMONY_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Lima"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                "days": {"type": "integer", "description": "Forecast days", "default": 1},
                "tags": {"type": "array", "items": {"type": "string"}},
                "window": {
                    "type": "object",
                    "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
                    "required": ["start"],
                },
                "detail": {
                    "oneOf": [{"type": "string", "description": "a named level"}, {"type": "number"}],
                    "default": "low",
                },
                "points": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"lat": {"type": "number"}}},
                    "nullable": True,
                },
                "kind": {"type": ["string", "null"]},
            },
            "required": ["city"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Current time.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

WEATHER_CALL = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lima", "units": "celsius"}}}

HARMONY_CASES = [
    ("user_only", {
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "add_generation_prompt": True,
    }),
    ("developer_identity_effort", {
        "messages": [
            {"role": "system", "content": "Answer in Spanish."},
            {"role": "user", "content": "Hi!"},
        ],
        "add_generation_prompt": True,
        "kwargs": {"reasoning_effort": "high", "model_identity": "You are a careful assistant."},
    }),
    ("history_drops_thinking", {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "thinking": "Add them.", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("training_final_turn", {
        "messages": [
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "thinking": "Simple sum.", "content": "6"},
        ],
        "add_generation_prompt": False,
    }),
    ("tools_and_calls", {
        "messages": [
            {"role": "user", "content": "Weather in Lima?"},
            {"role": "assistant", "thinking": "I need the weather tool.", "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "content": "{\"temp\": 19, \"sky\": \"clear\"}"},
            {"role": "assistant", "content": "It is 19 °C and clear."},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": HARMONY_TOOLS,
        "add_generation_prompt": True,
    }),
    ("pending_tool_call", {
        "messages": [
            {"role": "system", "content": "Use tools when they help."},
            {"role": "user", "content": "Weather in Lima?"},
            {"role": "assistant", "content": "Checking the weather.", "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "content": "sunny"},
        ],
        "tools": HARMONY_TOOLS,
        "add_generation_prompt": True,
    }),
    ("builtin_tools", {
        "messages": [{"role": "user", "content": "Find the Julia 1.12 release notes."}],
        "add_generation_prompt": True,
        "kwargs": {"builtin_tools": ["browser", "python"], "reasoning_effort": "low"},
    }),
]

TEMPLATES = [
    ("qwen3", "Qwen/Qwen3-0.6B", QWEN_CASES),
    ("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct", QWEN_CASES),
    ("gpt-oss", "openai/gpt-oss-20b", HARMONY_CASES),
]


def record(name, repo, case_list):
    tokenizer = AutoTokenizer.from_pretrained(repo)
    cases = []
    for case_name, case in case_list:
        kwargs = dict(case.get("kwargs", {}))
        pinned = {}
        if name == "gpt-oss":
            kwargs["strftime_now"] = lambda fmt: DATE
            pinned = {"date": DATE}
        rendered = tokenizer.apply_chat_template(
            case["messages"],
            tools=case.get("tools"),
            add_generation_prompt=case["add_generation_prompt"],
            tokenize=False,
            **kwargs,
        )
        cases.append({"name": case_name, **case, **pinned, "expected": rendered})
    print(f"{name}: {len(cases)} cases from {repo}")
    return {"name": name, "repo_id": repo, "template": tokenizer.chat_template, "cases": cases}


def main(names):
    fixture = {"templates": []}
    if names and os.path.exists(OUT):
        with open(OUT) as f:
            fixture = json.load(f)
    recorded = {t["name"]: t for t in fixture["templates"]}
    for name, repo, case_list in TEMPLATES:
        if not names or name in names:
            recorded[name] = record(name, repo, case_list)
    fixture["templates"] = [recorded[name] for name, _, _ in TEMPLATES if name in recorded]
    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1, ensure_ascii=False)
    print(f"Wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
