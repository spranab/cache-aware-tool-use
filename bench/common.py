from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("PROVIDER", "deepseek").lower()

_PROVIDER_DEFAULTS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "reasoner": "deepseek-chat",
        "broker": "deepseek-chat",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "reasoner": "gpt-4o",
        "broker": "gpt-4o-mini",
    },
}

if PROVIDER not in _PROVIDER_DEFAULTS:
    raise SystemExit(f"unknown PROVIDER: {PROVIDER}; expected one of {list(_PROVIDER_DEFAULTS)}")

_cfg = _PROVIDER_DEFAULTS[PROVIDER]
API_KEY = os.environ.get(_cfg["api_key_env"])
BASE_URL = os.environ.get("BASE_URL", _cfg["base_url"])
REASONER_MODEL = os.environ.get("REASONER_MODEL", _cfg["reasoner"])
BROKER_MODEL = os.environ.get("BROKER_MODEL", _cfg["broker"])

ROOT = Path(__file__).parent.parent
RUN_DIR = ROOT / "runs"
RUN_DIR.mkdir(exist_ok=True)


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class TestCase:
    case_id: str
    user_goal: str
    tools: list[ToolDef]
    expected_tool: str
    expected_args: dict[str, Any]


@dataclass
class ArmResult:
    case_id: str
    arm: str
    selected_tool: str | None
    selected_args: dict[str, Any] | None
    tool_selection_correct: bool
    arg_accuracy: float
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "arm": self.arm,
            "selected_tool": self.selected_tool,
            "selected_args": self.selected_args,
            "tool_selection_correct": self.tool_selection_correct,
            "arg_accuracy": self.arg_accuracy,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": self.latency_s,
            "error": self.error,
            "extra": self.extra,
        }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
