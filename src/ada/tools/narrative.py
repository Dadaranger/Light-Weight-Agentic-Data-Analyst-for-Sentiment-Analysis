"""6-element narrative extraction — Day 4 Module A.

LLM-driven when available, template-based fallback otherwise. The template
fallback produces "needs human" placeholders rather than fabricating
narratives, so the audit log makes it clear what was machine-extracted vs.
human-confirmed.
"""
from __future__ import annotations

import json
from typing import Any

from ada.llm.client import call_json


SIX_ELEMENT_KEYS = ("ACTOR", "ACTION", "VICTIM", "BLAME", "MORAL", "DESIRED")


def template_narrative(
    topic_label: str, keywords: list[str], samples: list[str], language: str = "zh-TW"
) -> dict[str, Any]:
    """Conservative template — surfaces what's verifiable, defers judgment.

    Use this when the LLM is unavailable. Every six-element field is marked as
    "needs human" so a downstream HITL or memory step doesn't silently treat
    it as machine output.
    """
    keyword_str = "·".join(keywords[:5]) if keywords else "—"
    is_zh = language.startswith("zh")
    todo = "（待人工判定）" if is_zh else "(awaiting human judgment)"
    if is_zh:
        narrative = (
            f"關於「{topic_label}」的討論集中於關鍵詞「{keyword_str}」。"
            f"具體敘事框架（行為者、責任、預期反應）需要分析師根據樣本內容進一步判斷。"
        )
    else:
        narrative = (
            f"Discussion in topic '{topic_label}' clusters around keywords '{keyword_str}'. "
            "Six-element narrative framing requires analyst judgment from sample posts."
        )
    return {
        "ACTOR": todo,
        "ACTION": todo,
        "VICTIM": todo,
        "BLAME": todo,
        "MORAL": todo,
        "DESIRED": todo,
        "NARRATIVE_STATEMENT": narrative,
        "_source": "template",
    }


_LLM_PROMPT = """\
You are extracting a six-element narrative from a topic cluster's keywords + sample posts.

Topic label: {topic_label}
Top keywords: {keywords}
Sample posts ({n_samples}):
{samples_block}

Return ONE JSON object with these fields:
{{
  "ACTOR": "<who is the protagonist or blamed party>",
  "ACTION": "<what they did or failed to do>",
  "VICTIM": "<who suffered or is affected>",
  "BLAME": "<why this happened>",
  "MORAL": "<framing as right/wrong>",
  "DESIRED": "<what reaction the narrative wants from the audience>",
  "NARRATIVE_STATEMENT": "<one-sentence narrative summary, ≤ 60 words>"
}}

Rules:
- Use the language of the source posts ({language}).
- If a field is genuinely unclear from the samples, write "（不明確）" / "(unclear)".
- Do NOT fabricate actors or victims that the samples don't support.
- The NARRATIVE_STATEMENT is what would appear in the analytic brief — make it precise.

Return JSON only.
"""


def llm_narrative(
    topic_label: str, keywords: list[str], samples: list[str], language: str = "zh-TW"
) -> dict[str, Any]:
    """Extract via planner LLM. Caller is responsible for catching exceptions."""
    samples_block = "\n".join(f"  - {s[:200]}" for s in samples[:8])
    raw = call_json(
        "planner",
        "_inline_narrative",
        # render via call_json's render_template → no template file lookup needed
        # because we'll raise if the template doesn't exist; instead use direct prompt
    )
    # call_json signature requires a prompt template name. To avoid creating a
    # new template file just for this, we'll inline the prompt below via a
    # one-off call.
    return raw  # unreachable — see _llm_narrative_inline below


def _llm_narrative_inline(
    topic_label: str, keywords: list[str], samples: list[str], language: str
) -> dict[str, Any]:
    """LLM call that builds the prompt inline (avoids creating a template file)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from ada.llm.client import _coerce_json, _json_model

    samples_block = "\n".join(f"  - {s[:200]}" for s in samples[:8])
    prompt = _LLM_PROMPT.format(
        topic_label=topic_label,
        keywords=", ".join(keywords[:10]),
        n_samples=min(len(samples), 8),
        samples_block=samples_block,
        language=language,
    )
    msg = _json_model("planner").invoke([
        SystemMessage(content="Return JSON only. No prose."),
        HumanMessage(content=prompt),
    ])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    out = _coerce_json(content)
    if isinstance(out, list):
        out = out[0] if out else {}
    out["_source"] = "llm"
    return out


def extract_narrative(
    topic_label: str, keywords: list[str], samples: list[str], language: str = "zh-TW"
) -> dict[str, Any]:
    """Try LLM, fall back to template. Always returns a 6-element dict."""
    try:
        result = _llm_narrative_inline(topic_label, keywords, samples, language)
        # Ensure all required keys present
        for k in SIX_ELEMENT_KEYS:
            result.setdefault(k, "")
        result.setdefault("NARRATIVE_STATEMENT", "")
        return result
    except Exception as e:
        fallback = template_narrative(topic_label, keywords, samples, language)
        fallback["_llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return fallback
