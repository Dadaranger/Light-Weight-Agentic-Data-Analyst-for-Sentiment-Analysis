"""Clean stage — port of Day 1 cells 22-24.

Three-stream split:
  - organic    : real users, valid text, no dup
  - quarantine : empty/null text (kept as separate parquet, not deleted)
  - bots       : suspected automated accounts (kept for amplification analysis)

Audit log records every removal/split with affected row counts and reason.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ada.config import settings
from ada.state import (
    AuditEntry,
    GraphState,
    Stage,
    StageArtifact,
)
from ada.tools.hashing import hash_file


# Author values that mean "automated/suspected bot" across the languages
# we support. Domain memory can override via `domain_knowledge.notes`
# entries in future slices.
BOT_LABELS = {
    "疑似機器人",       # zh-TW (course material)
    "可疑機器人",
    "机器人",            # zh-CN
    "suspected_bot",
    "bot",
}


def _clean_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "clean" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(df: pd.DataFrame, path: Path) -> tuple[Path, str]:
    df.to_parquet(path, index=False)
    return path, hash_file(path)


def clean_node(state: GraphState) -> dict:
    if state.canonical_data_path is None:
        raise RuntimeError("clean requires a canonical parquet from reshape")

    df = pd.read_parquet(Path(state.canonical_data_path)).copy()
    starting_rows = len(df)
    out_dir = _clean_dir(state)
    audit_entries: list[AuditEntry] = list(state.audit_log)

    def _audit(action: str, affected: int, reason: str, path: Path | None = None, h: str | None = None) -> None:
        audit_entries.append(AuditEntry(
            timestamp=datetime.now(timezone.utc),
            stage=Stage.CLEAN,
            action=action,
            affected_rows=affected,
            reason=reason,
            artifact_path=str(path) if path else None,
            artifact_hash=h,
        ))

    # ── Step 1: composite-key dedup (platform + id) ──────────────────────────
    if "platform" in df.columns and "id" in df.columns:
        df["_composite"] = df["platform"].astype(str) + "_" + df["id"].astype(str)
        before = len(df)
        df = df.drop_duplicates(subset="_composite", keep="first")
        removed = before - len(df)
        df = df.drop(columns="_composite")
        _audit("removed composite-key duplicates", removed,
               "platform+id appeared more than once; kept first")

    # ── Step 2: quarantine empty/null text ───────────────────────────────────
    valid_text = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
    quarantine_df = df.loc[~valid_text].copy()
    df = df.loc[valid_text].copy()
    if not quarantine_df.empty:
        q_path, q_hash = _write(quarantine_df, out_dir / "quarantine.parquet")
        _audit("quarantined empty/null text", len(quarantine_df),
               "preserved separately, not deleted", q_path, q_hash)
    else:
        q_path, q_hash = None, None

    # ── Step 3: split bots vs organic ────────────────────────────────────────
    bot_path, bot_hash = None, None
    if "author" in df.columns:
        bot_mask = df["author"].astype(str).isin(BOT_LABELS)
        bot_df = df.loc[bot_mask].copy()
        df = df.loc[~bot_mask].copy()
        if not bot_df.empty:
            bot_path, bot_hash = _write(bot_df, out_dir / "bots.parquet")
            _audit("split bot-labeled accounts to bot stream", len(bot_df),
                   f"matched author labels: {sorted(set(bot_df['author']))}",
                   bot_path, bot_hash)

    # ── Step 4: write organic stream ─────────────────────────────────────────
    organic_path, organic_hash = _write(df, out_dir / "organic.parquet")
    _audit("wrote organic stream", len(df),
           "remaining rows after dedup + quarantine + bot split",
           organic_path, organic_hash)

    # ── Stage artifact ──────────────────────────────────────────────────────
    bot_pct = (
        round((len(df) and (1 - len(df) / starting_rows)) * 100, 2)
        if starting_rows else 0.0
    )
    summary = {
        "starting_rows": starting_rows,
        "organic_rows": int(len(df)),
        "quarantine_rows": int(len(quarantine_df)),
        "bot_rows": int(0 if bot_path is None else (
            pd.read_parquet(bot_path).shape[0] if bot_path else 0
        )),
        "organic_pct": round(len(df) / starting_rows * 100, 2) if starting_rows else 0.0,
        "streams": {
            "organic": str(organic_path),
            "quarantine": str(q_path) if q_path else None,
            "bots": str(bot_path) if bot_path else None,
        },
    }
    # Cross-check bot threshold against domain memory
    bot_share = summary["bot_rows"] / starting_rows * 100 if starting_rows else 0
    threshold = state.domain_knowledge.thresholds.bot_quarantine_pct
    if bot_share > threshold:
        summary["bot_alarm"] = (
            f"bot share {bot_share:.1f}% exceeds threshold {threshold}%"
        )

    artifact = StageArtifact(
        stage=Stage.CLEAN,
        parquet_path=str(organic_path),  # downstream stages should read this
        parquet_hash=organic_hash,
        summary_stats=summary,
        notes=(
            f"organic={summary['organic_rows']:,}, "
            f"quarantine={summary['quarantine_rows']:,}, "
            f"bots={summary['bot_rows']:,}"
        ),
    )

    return {
        "artifacts": {**state.artifacts, Stage.CLEAN: artifact},
        "audit_log": audit_entries,
        "completed_stages": [*state.completed_stages, Stage.CLEAN],
    }
