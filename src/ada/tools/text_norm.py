"""Text normalization — strips URLs, mentions, hashtag symbols, and noise.

Mirrors Day 2's `text_norm` cell. Conservative: keep punctuation and content;
only remove things that are pure noise for sentiment/topic analysis.
"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"[@＠][\w一-鿿]+")
_HASHTAG_SYMBOL = re.compile(r"[#＃]")
_REPEATED_PUNCT = re.compile(r"([!?。！？.])\1{2,}")
_WHITESPACE = re.compile(r"\s+")
_RT_PREFIX = re.compile(r"^(?:RT|rt|轉發|轉貼|分享):?\s*", re.IGNORECASE)


def normalize(text: str) -> str:
    """Return text with URLs/mentions stripped and whitespace collapsed.

    Empty / non-string inputs return empty string. Hashtag symbol is removed
    but the hashtag word is preserved (e.g. `#颱風` → `颱風`).
    """
    if not isinstance(text, str) or not text:
        return ""
    s = text
    s = _RT_PREFIX.sub("", s)
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = _HASHTAG_SYMBOL.sub("", s)
    s = _REPEATED_PUNCT.sub(r"\1\1", s)  # cap repeats at 2
    s = _WHITESPACE.sub(" ", s).strip()
    return s
