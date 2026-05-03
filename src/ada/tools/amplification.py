"""Amplification proxy indicators — Day 4 Module B.

Four proxies plus a coordination-signal checklist. All deterministic
(no LLM). Outputs are JSON-serializable summary dicts.

Important: these are *proxies*, not measurements. The agent has only the
data the dataset provides — no follower graph, no cross-platform identity,
no account history. Findings must always be reported with this limitation.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Proxy 01 — engagement concentration
# ─────────────────────────────────────────────────────────────────────────────

def proxy_engagement_concentration(df: pd.DataFrame) -> dict[str, Any]:
    if "engagement" not in df.columns:
        return {"available": False}
    eng = pd.to_numeric(df["engagement"], errors="coerce").dropna()
    if eng.empty or eng.sum() == 0:
        return {"available": False, "reason": "no engagement signal"}
    n_top5 = max(1, int(len(eng) * 0.05))
    top5_sum = float(eng.nlargest(n_top5).sum())
    total = float(eng.sum())
    return {
        "available": True,
        "n_total": int(len(eng)),
        "n_top5pct": n_top5,
        "top5_share_pct": round(top5_sum / total * 100, 2),
        "median_engagement": float(eng.median()),
        "max_engagement": float(eng.max()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proxy 02 — author-type concentration per topic
# ─────────────────────────────────────────────────────────────────────────────

def proxy_author_concentration(
    df: pd.DataFrame, topic_col: str = "analyst_label", threshold_pct: float = 70.0
) -> dict[str, Any]:
    if "author" not in df.columns or topic_col not in df.columns:
        return {"available": False}
    cross = (
        pd.crosstab(df[topic_col], df["author"], normalize="index").mul(100).round(1)
    )
    if cross.empty:
        return {"available": False, "reason": "no rows in cross-tab"}
    flagged: list[dict] = []
    for label, row in cross.iterrows():
        max_pct = float(row.max())
        if max_pct >= threshold_pct:
            flagged.append({
                "topic": str(label),
                "dominant_author": str(row.idxmax()),
                "pct": max_pct,
            })
    return {
        "available": True,
        "topic_count": int(len(cross)),
        "concentrated_topics": flagged,
        "concentration_threshold_pct": threshold_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proxy 03 — temporal bursts (3σ above mean)
# ─────────────────────────────────────────────────────────────────────────────

def proxy_temporal_bursts(df: pd.DataFrame, sigma: float = 3.0) -> dict[str, Any]:
    if "ts" not in df.columns:
        return {"available": False}
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"available": False, "reason": "no parseable timestamps"}
    hourly = ts.dt.floor("h").value_counts().sort_index()
    if len(hourly) < 5:
        return {"available": False, "reason": "fewer than 5 distinct hours"}
    mean = float(hourly.mean())
    std = float(hourly.std()) if len(hourly) > 1 else 0.0
    threshold = mean + sigma * std
    bursts = hourly[hourly > threshold]
    return {
        "available": True,
        "mean_per_hour": round(mean, 2),
        "std_per_hour": round(std, 2),
        "sigma": sigma,
        "threshold": round(threshold, 2),
        "burst_count": int(len(bursts)),
        "bursts": [
            {"timestamp": ts.isoformat(), "count": int(c)}
            for ts, c in bursts.items()
        ][:10],  # cap for JSON readability
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proxy 04 — content duplication
# ─────────────────────────────────────────────────────────────────────────────

def proxy_content_duplication(df: pd.DataFrame) -> dict[str, Any]:
    if "text" not in df.columns:
        return {"available": False}
    valid = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
    sub = df.loc[valid]
    n = int(len(sub))
    if n == 0:
        return {"available": False, "reason": "no valid text"}
    dup = sub.duplicated(subset="text", keep=False)
    dup_count = int(dup.sum())
    # Top 5 most-repeated texts
    most_repeated = sub.loc[dup, "text"].value_counts().head(5)
    return {
        "available": True,
        "n_valid": n,
        "duplicate_count": dup_count,
        "duplicate_pct": round(dup_count / n * 100, 2),
        "top_repeated": [
            {"text": str(t)[:80], "count": int(c)}
            for t, c in most_repeated.items()
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coordination signal checklist
# ─────────────────────────────────────────────────────────────────────────────

def coordination_signals(
    p1: dict, p2: dict, p3: dict, p4: dict,
    bot_share_pct: float,
    duplication_alarm_pct: float = 20.0,
    bot_quarantine_pct: float = 10.0,
    engagement_concentration_alarm: float = 50.0,
) -> dict[str, Any]:
    """Six-signal checklist matching Day 4 Slide 12."""
    signals: list[dict] = []

    if p4.get("available"):
        dup_pct = p4["duplicate_pct"]
        signals.append({
            "name": "high content duplication",
            "triggered": dup_pct > duplication_alarm_pct,
            "evidence": f"{dup_pct}% duplicate rate (threshold {duplication_alarm_pct}%)",
        })

    signals.append({
        "name": "bot accounts present at scale",
        "triggered": bot_share_pct > bot_quarantine_pct,
        "evidence": f"bot share {bot_share_pct:.2f}% (threshold {bot_quarantine_pct}%)",
    })

    signals.append({
        "name": "regular timing intervals (account-level)",
        "triggered": False,
        "evidence": "requires account-level timestamp data; not available in this dataset",
    })

    signals.append({
        "name": "cross-platform synchrony",
        "triggered": False,
        "evidence": "requires cross-platform identity resolution; not available",
    })

    if p3.get("available"):
        bursts = p3["burst_count"]
        signals.append({
            "name": "temporal bursts above 3σ",
            "triggered": bursts > 0,
            "evidence": f"{bursts} bursts identified",
        })

    if p1.get("available"):
        top5 = p1["top5_share_pct"]
        signals.append({
            "name": "engagement concentration",
            "triggered": top5 > engagement_concentration_alarm,
            "evidence": f"top 5% carry {top5}% of total engagement",
        })

    triggered_count = sum(1 for s in signals if s["triggered"])
    available_count = sum(1 for s in signals if "not available" not in s["evidence"])

    if triggered_count >= 4:
        confidence = "HIGH"
        reason = "multiple independent signals point to coordination"
    elif triggered_count >= 2:
        confidence = "MODERATE"
        reason = "some signals present; data gaps prevent stronger conclusion"
    else:
        confidence = "LOW"
        reason = "insufficient signals to assert coordination"

    return {
        "signals": signals,
        "triggered_count": triggered_count,
        "available_count": available_count,
        "total_count": len(signals),
        "coordination_confidence": confidence,
        "confidence_reason": reason,
    }
