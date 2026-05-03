"""EDA statistics — mirrors the calculations from Day 1 cells 17-20.

All functions take a canonical-schema DataFrame (columns: id, text, ts,
author, engagement, platform — any of the optional ones may be missing)
and return JSON-serializable summaries.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def temporal_summary(df: pd.DataFrame) -> dict[str, Any]:
    if "ts" not in df.columns:
        return {"available": False}
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return {"available": False, "reason": "no parseable timestamps"}
    duration_days = (ts.max() - ts.min()).total_seconds() / 86400
    by_6h = (
        df.assign(_ts=pd.to_datetime(df["ts"], utc=True))
        .dropna(subset=["_ts"])
        .set_index("_ts")
        .resample("6h")["id"]
        .count()
    )
    peak_idx = by_6h.idxmax() if not by_6h.empty else None
    return {
        "available": True,
        "start": ts.min().isoformat(),
        "end": ts.max().isoformat(),
        "duration_days": round(duration_days, 2),
        "peak_6h_window": peak_idx.isoformat() if peak_idx is not None else None,
        "peak_6h_count": int(by_6h.max()) if not by_6h.empty else 0,
    }


def categorical_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {"available": False}
    s = df[col].astype(str)
    counts = s.value_counts()
    pct = s.value_counts(normalize=True).mul(100).round(2)
    return {
        "available": True,
        "n_unique": int(s.nunique()),
        "top": counts.head(10).to_dict(),
        "top_pct": pct.head(10).to_dict(),
    }


def engagement_summary(df: pd.DataFrame) -> dict[str, Any]:
    if "engagement" not in df.columns:
        return {"available": False}
    eng = pd.to_numeric(df["engagement"], errors="coerce").dropna()
    if eng.empty:
        return {"available": False, "reason": "no numeric engagement values"}
    return {
        "available": True,
        "count": int(len(eng)),
        "median": float(eng.median()),
        "mean": float(eng.mean()),
        "max": float(eng.max()),
        "p99": float(eng.quantile(0.99)),
        "zero_count": int((eng == 0).sum()),
        "top5_share_pct": round(
            eng.nlargest(max(1, int(len(eng) * 0.05))).sum() / eng.sum() * 100, 2
        ),
    }


def text_length_summary(df: pd.DataFrame) -> dict[str, Any]:
    if "text" not in df.columns:
        return {"available": False}
    lens = df["text"].fillna("").astype(str).str.len()
    return {
        "available": True,
        "median": int(lens.median()),
        "mean": round(float(lens.mean()), 1),
        "max": int(lens.max()),
        "very_short_lt5": int((lens < 5).sum()),
        "very_long_gt200": int((lens > 200).sum()),
    }


def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Pre-clean signals: dedup candidates, empty text, future timestamps."""
    out: dict[str, Any] = {}

    # Composite-key duplicates (platform + id)
    if "platform" in df.columns and "id" in df.columns:
        composite = df["platform"].astype(str) + "_" + df["id"].astype(str)
        out["composite_dup_count"] = int(composite.duplicated().sum())
    else:
        out["composite_dup_count"] = int(df["id"].astype(str).duplicated().sum()) if "id" in df.columns else 0

    # Content duplicates (same text across different IDs)
    if "text" in df.columns:
        valid = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
        out["content_dup_count"] = int(df.loc[valid, "text"].duplicated(keep=False).sum())
        out["empty_text_count"] = int((~valid).sum())
    else:
        out["content_dup_count"] = 0
        out["empty_text_count"] = 0

    # Future timestamps
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        out["future_ts_count"] = int((ts > pd.Timestamp.now(tz="UTC")).sum())
    else:
        out["future_ts_count"] = 0

    return out
