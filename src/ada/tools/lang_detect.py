"""Language detection — wraps langdetect, with a Chinese-script shortcut.

Returns a normalized BCP-47 tag. We don't try to disambiguate zh-TW vs. zh-CN
purely from the language detector (it isn't reliable for short posts) — that's
the schema agent's job, using script + content heuristics.
"""
from __future__ import annotations

import re

_HAS_CJK = re.compile(r"[一-鿿]")
_TRAD_HINT = re.compile(r"[颱這個們麼說發點時間實國灣為來]")


def detect_one(text: str, default: str = "auto") -> str:
    """Detect language for a single string. Cheap, returns BCP-47 tag."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return default
    if _HAS_CJK.search(text):
        return "zh-TW" if _TRAD_HINT.search(text) else "zh"
    try:
        from langdetect import detect
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        return default
    try:
        return detect(text)
    except Exception:
        return default
