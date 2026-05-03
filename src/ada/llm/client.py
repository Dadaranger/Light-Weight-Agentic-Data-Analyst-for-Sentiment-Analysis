"""Thin Ollama wrapper. Two model tiers: `planner` (7B) and `router` (3B).

Planner calls return constrained JSON via Ollama's `format=json` mode.
Prompts are loaded from `ada/llm/prompts/*.md`. Variable substitution uses
explicit `replace` over a known var dict — NOT `str.format()` — so JSON
example blocks in the prompt templates stay intact.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ada.config import settings


@lru_cache(maxsize=2)
def _json_model(tier: str) -> ChatOllama:
    name = settings.planner_model if tier == "planner" else settings.router_model
    return ChatOllama(
        model=name,
        base_url=settings.ollama_host,
        format="json",
        temperature=0.2,
    )


@lru_cache(maxsize=2)
def _text_model(tier: str) -> ChatOllama:
    name = settings.planner_model if tier == "planner" else settings.router_model
    return ChatOllama(
        model=name,
        base_url=settings.ollama_host,
        temperature=0.3,
    )


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    return resources.files("ada.llm.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def render_template(template: str, vars: dict[str, Any]) -> str:
    """Replace `{var}` placeholders only for keys present in `vars`.

    Unlike `str.format()`, leaves unknown `{...}` (e.g. JSON examples) alone.
    """
    out = template
    for key, value in vars.items():
        out = out.replace("{" + key + "}", str(value))
    return out


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _coerce_json(text: str) -> dict | list:
    """Parse JSON, with a fallback that extracts the first JSON-shaped block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def call_json(tier: str, prompt_name: str, **vars: Any) -> dict | list:
    """Call the LLM with a templated prompt; return parsed JSON."""
    rendered = render_template(load_prompt(prompt_name), vars)
    msg = _json_model(tier).invoke([
        SystemMessage(content="Return JSON only. No prose, no code fences."),
        HumanMessage(content=rendered),
    ])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return _coerce_json(content)


def call_text(tier: str, prompt_name: str, **vars: Any) -> str:
    """Call the LLM and return raw text (for the brief writer)."""
    rendered = render_template(load_prompt(prompt_name), vars)
    msg = _text_model(tier).invoke([HumanMessage(content=rendered)])
    return msg.content if isinstance(msg.content, str) else str(msg.content)
