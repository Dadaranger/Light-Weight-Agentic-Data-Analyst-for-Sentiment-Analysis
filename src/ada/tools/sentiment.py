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
    r"(救[命我]|急救|被困|受傷|失蹤|危險|危急|崩潰|絕望|快不行|撐不住|快死|"
    r"活不下去|快淹|快倒|119打不|打不通|"
    r"求救|求援|求助|求求|拜託.*幫|幫幫忙)"
)
_DISTRESS_EN = re.compile(
    r"\b(help me|trapped|injured|missing|critical|dying|can't survive|emergency|sos)\b",
    re.IGNORECASE,
)

# Sarcasm / uncertainty markers — ambiguous polarity, flag for human review
_SARCASM_ZH = re.compile(
    r"(感謝.*(政府|官員|長官)|英明|偉大的(?:政府|決策)|呵呵.*(政府|官員)|果然.*厲害)"
)
_SARCASM_EN = re.compile(
    r"(thanks?\s+(?:government|gov|officials),?\s+very|so\s+much\s+for)", re.IGNORECASE
)

# Civic-discourse anger patterns — accountability rhetoric, direct rage,
# "look at the result" indictments. Lexicon misses these because the
# individual words aren't sentiment-bearing on their own.
_NEGATIVE_RHETORIC_ZH = re.compile(
    r"(誰[來要負]責|誰賠|怎麼[會還能]這樣|氣死人?|看[看一]下結果|當初.*現在|"
    r"預算被[刪砍]|經費被[刪砍]|砍預算|砍經費|根本(?:不|沒|是個)|"
    r"還在[等睡]|又(?:是|來)|早就說|早就警告)"
)
_POSITIVE_RHETORIC_ZH = re.compile(
    r"(終於|總算|還好|幸好|多虧|要不是|多謝)"
)

# Official news / announcement framing — content may include negative trigger
# words ("警報", "災害") but the post itself is informational, not an opinion.
# Heuristic: starts with 【...】 bracket header containing media/agency cues.
_ANNOUNCEMENT_ZH = re.compile(
    r"^[\s]*[【\[](?:[^】\]]{0,15})"
    r"(?:政府|市府|縣府|公所|新聞|快訊|直播|現場|公告|通知|警報|氣象|消防|警察)"
    r"[^】\]]{0,15}[】\]]"
)


def tier1_rules(text: str, language: str) -> tuple[Label, float]:
    """Return (label, confidence). Confidence in [0, 1].

    Default NEUTRAL has very low confidence (0.05) — it means "no rule fired",
    not "I assert this is neutral". The combiner will defer to T2 in that case.
    """
    if not text:
        return "UNCERTAIN", 0.0

    is_zh = language.startswith("zh")
    distress_re = _DISTRESS_ZH if is_zh else _DISTRESS_EN
    sarcasm_re = _SARCASM_ZH if is_zh else _SARCASM_EN

    # Order matters: ambiguity > distress > announcement > rhetoric > default
    if sarcasm_re.search(text):
        return "UNCERTAIN", 0.55
    if distress_re.search(text):
        return "NEGATIVE-DISTRESS", 0.85
    if is_zh and _ANNOUNCEMENT_ZH.search(text):
        # News/announcement framing — assert NEUTRAL with high enough confidence
        # that the lexicon can't override informational content
        return "NEUTRAL", 0.7
    if is_zh and _NEGATIVE_RHETORIC_ZH.search(text):
        return "NEGATIVE", 0.65
    if is_zh and _POSITIVE_RHETORIC_ZH.search(text):
        return "POSITIVE", 0.6
    return "NEUTRAL", 0.05  # "no rule fired" — defer to T2 lexicon


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

    `agreed` means "the label is well-determined; the human shouldn't need to
    review this row" — NOT "the two tiers produced identical strings".

    A row is *not* agreed only when:
      - Either tier returned UNCERTAIN (sarcasm / genuine ambiguity), or
      - Both tiers had opinions and they conflict on polarity

    Rules (in order):
    1. UNCERTAIN from either tier wins — sarcasm / ambiguity flag is sticky
    2. NEGATIVE-DISTRESS from T1 wins — urgency rule is high-stakes; T2 has no
       distress vocabulary so a NEUTRAL T2 isn't disagreement
    3. NEUTRAL is "no opinion" — defer to whichever tier has signal; this is
       complementary, not conflicting
    4. Same polarity → pick the more specific label (DISTRESS > NEGATIVE)
    5. True polarity disagreement → defer to higher confidence by ≥0.15,
       else UNCERTAIN; mark as not agreed
    """
    # 1. Sarcasm / ambiguity → flag for review
    if t1 == "UNCERTAIN" or t2 == "UNCERTAIN":
        return "UNCERTAIN", False

    # 2. T1 distress rule fires — T2 absence isn't disagreement
    if t1 == "NEGATIVE-DISTRESS":
        return t1, True

    # 3. NEUTRAL = no signal → defer; this is complementary, not disagreement.
    #    BUT a high-confidence NEUTRAL (e.g. announcement detection) is an
    #    assertion, not abstention — it should hold against weak T2 lexicon hits.
    if t1 == "NEUTRAL" and t2 != "NEUTRAL":
        if t1_conf >= 0.5 and t2_conf < t1_conf:
            return t1, True  # asserted neutral wins
        return t2, True
    if t2 == "NEUTRAL" and t1 != "NEUTRAL":
        if t2_conf >= 0.5 and t1_conf < t2_conf:
            return t2, True
        return t1, True

    # 4 & 5. Both opinionated, or both NEUTRAL
    polarity = lambda lbl: 1 if lbl == "POSITIVE" else -1 if lbl.startswith("NEGATIVE") else 0
    p1, p2 = polarity(t1), polarity(t2)
    if p1 == p2:
        return (
            "NEGATIVE-DISTRESS" if "DISTRESS" in (t1, t2) else (t1 if t1_conf >= t2_conf else t2)
        ), True
    # True conflict
    winner = t1 if t1_conf > t2_conf + 0.15 else t2 if t2_conf > t1_conf + 0.15 else "UNCERTAIN"
    return winner, False
