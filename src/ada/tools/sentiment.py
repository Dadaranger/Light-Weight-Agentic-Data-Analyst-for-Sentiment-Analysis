"""Three-tier sentiment baseline (Tiers 1 + 2 implemented in Slice 3).

Tier 1 — Rules: keyword/regex patterns produce a coarse label. Fast, opinionated,
        misses sarcasm.
Tier 2 — Lexicon: weighted sentiment dictionary with negator + intensifier
        handling. Confirms or refines T1.
Tier 3 — Transformer: TBD (later slice). Will use sentence-transformers or an
        Ollama-served classifier model.

Labels: POSITIVE | NEGATIVE | NEUTRAL | NEGATIVE-DISTRESS | UNCERTAIN
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Iterable, Literal

import yaml

Label = Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE-DISTRESS", "UNCERTAIN"]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — rules
# ─────────────────────────────────────────────────────────────────────────────

# Distress / urgency markers — when present alongside negative content, escalate
# from NEGATIVE → NEGATIVE-DISTRESS. Mirrors Day 2's three-tier escalation.
_DISTRESS_ZH = re.compile(
    r"(救[命我]|被困|受傷|失蹤|危險|崩潰|絕望|快不行|撐不住|快死|活不下去)"
)
_DISTRESS_EN = re.compile(
    r"\b(help me|trapped|injured|missing|critical|dying|can't survive|emergency)\b",
    re.IGNORECASE,
)

# Sarcasm / uncertainty markers — ambiguous polarity, flag for human review
_SARCASM_ZH = re.compile(r"(感謝.*(政府|官員|長官)|英明|偉大的(?:政府|決策))")
_SARCASM_EN = re.compile(r"(thanks?\s+(?:government|gov|officials),?\s+very)", re.IGNORECASE)


def tier1_rules(text: str, language: str) -> tuple[Label, float]:
    """Return (label, confidence). Confidence in [0, 1]."""
    if not text:
        return "UNCERTAIN", 0.0

    is_zh = language.startswith("zh")
    distress_re = _DISTRESS_ZH if is_zh else _DISTRESS_EN
    sarcasm_re = _SARCASM_ZH if is_zh else _SARCASM_EN

    if sarcasm_re.search(text):
        return "UNCERTAIN", 0.5
    if distress_re.search(text):
        return "NEGATIVE-DISTRESS", 0.85
    return "NEUTRAL", 0.4  # default; T2 lexicon will sharpen this


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — lexicon
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _lexicons() -> dict:
    return yaml.safe_load(
        resources.files("ada.memory.seeds")
        .joinpath("sentiment_lexicons.yaml")
        .read_text(encoding="utf-8")
    ) or {}


def _lex_for(language: str) -> tuple[dict[str, int], dict[str, int], set[str], set[str]]:
    lex = _lexicons()
    # Language fallback chain: zh-TW → zh → en (last resort)
    base = lex.get(language, {})
    if language.startswith("zh") and not base:
        base = lex.get("zh", {}) or lex.get("zh-TW", {})
    pos = {w: int(p) for w, p in base.get("positive", [])}
    neg = {w: int(p) for w, p in base.get("negative", [])}
    intens = {w: int(v) for w, v in base.get("intensifiers", [])}
    negators = set(base.get("negators", []))
    return pos, neg, intens, negators


def tier2_lexicon(tokens: Iterable[str], language: str) -> tuple[Label, float, int]:
    """Return (label, confidence, raw_score). Tokens come from `tokenize.tokenize`."""
    pos, neg, intens, negators = _lex_for(language)
    if not pos and not neg:
        return "UNCERTAIN", 0.0, 0

    score = 0
    hits = 0
    toks = list(tokens)
    for i, t in enumerate(toks):
        weight = pos.get(t, 0) + neg.get(t, 0)
        if weight == 0:
            continue
        # Look back 2 tokens for a negator → flip
        flipped = any(toks[j] in negators for j in range(max(0, i - 2), i))
        if flipped:
            weight = -weight
        score += weight
        hits += 1

    if hits == 0:
        return "NEUTRAL", 0.3, 0

    confidence = min(1.0, hits / 5)  # more hits → more confidence, capped
    if score >= 2:
        return "POSITIVE", confidence, score
    if score <= -3:
        return "NEGATIVE-DISTRESS" if score <= -5 else "NEGATIVE", confidence, score
    if score <= -1:
        return "NEGATIVE", confidence, score
    if score >= 1:
        return "POSITIVE", confidence, score
    return "NEUTRAL", confidence * 0.7, score


# ─────────────────────────────────────────────────────────────────────────────
# Combiner — picks the final label from T1 + T2
# ─────────────────────────────────────────────────────────────────────────────

def combine(t1: Label, t2: Label, t1_conf: float, t2_conf: float) -> tuple[Label, bool]:
    """Return (final_label, agreed).

    Rules:
    - UNCERTAIN from either tier sticks (sarcasm / ambiguity flagged)
    - DISTRESS from T1 outweighs T2 (urgency is high-stakes)
    - Otherwise prefer the more confident tier
    - If they disagree on polarity, escalate via UNCERTAIN
    """
    if t1 == "UNCERTAIN" or t2 == "UNCERTAIN":
        return "UNCERTAIN", False
    if t1 == "NEGATIVE-DISTRESS":
        return t1, t2.startswith("NEGATIVE")

    polarity = lambda lbl: 1 if lbl == "POSITIVE" else -1 if lbl.startswith("NEGATIVE") else 0
    p1, p2 = polarity(t1), polarity(t2)
    if p1 == p2:
        # Same polarity — pick the more specific (DISTRESS over plain NEGATIVE)
        return ("NEGATIVE-DISTRESS" if "DISTRESS" in (t1, t2) else (t1 if t1_conf >= t2_conf else t2)), True
    # Disagreement — defer to higher confidence; flag as not agreed
    winner = t1 if t1_conf > t2_conf + 0.15 else t2 if t2_conf > t1_conf + 0.15 else "UNCERTAIN"
    return winner, False
