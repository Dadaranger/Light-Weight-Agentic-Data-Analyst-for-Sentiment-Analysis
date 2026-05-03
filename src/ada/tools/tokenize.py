"""Language-aware tokenization. Jieba for zh, regex for en/other.

Returns a list of tokens. Stopword filtering happens here too — pulled from
`memory/seeds/language_norms.yaml`. Domain-specific extras can be added by
mutating `EXTRA_STOPWORDS` (Slice 5 will wire that to domain memory).
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Iterable

import yaml

_WORD_EN = re.compile(r"[A-Za-z']+")
_PUNCT_TRAILING = re.compile(r"^\W+|\W+$")

EXTRA_STOPWORDS: dict[str, set[str]] = {}  # populated by domain memory in Slice 5


@lru_cache(maxsize=1)
def _language_norms() -> dict:
    return yaml.safe_load(
        resources.files("ada.memory.seeds")
        .joinpath("language_norms.yaml")
        .read_text(encoding="utf-8")
    ) or {}


@lru_cache(maxsize=1)
def _jieba():
    import jieba
    # Load Taiwan-specific extras if present in the seed file
    norms = _language_norms()
    extras = norms.get("zh-TW", {}).get("jieba_dict_extras", [])
    for entry in extras:
        if isinstance(entry, list) and len(entry) >= 1:
            word = entry[0]
            freq = entry[1] if len(entry) > 1 else None
            tag = entry[2] if len(entry) > 2 else None
            try:
                jieba.add_word(word, freq=freq, tag=tag)
            except Exception:  # noqa: BLE001
                pass
    return jieba


def _stopwords_for(language: str) -> set[str]:
    norms = _language_norms()
    base = set(norms.get(language, {}).get("stopwords_extra", []))
    base |= EXTRA_STOPWORDS.get(language, set())
    # Also include the broader zh stopwords for any zh-* tag
    if language.startswith("zh") and language != "zh":
        base |= set(norms.get("zh", {}).get("stopwords_extra", []))
    return base


def tokenize(text: str, language: str) -> list[str]:
    """Return a list of meaningful tokens for the given language.

    Filters stopwords, single-char punctuation, and tokens that are pure
    digits or whitespace.
    """
    if not isinstance(text, str) or not text:
        return []

    if language.startswith("zh"):
        toks = list(_jieba().cut(text, cut_all=False))
    else:
        toks = _WORD_EN.findall(text.lower())

    stop = _stopwords_for(language)
    out: list[str] = []
    for t in toks:
        t = _PUNCT_TRAILING.sub("", t).strip()
        if not t or t.isspace():
            continue
        if t in stop:
            continue
        if len(t) == 1 and not t.isalnum():
            continue  # standalone punctuation
        if t.isdigit():
            continue
        out.append(t)
    return out
