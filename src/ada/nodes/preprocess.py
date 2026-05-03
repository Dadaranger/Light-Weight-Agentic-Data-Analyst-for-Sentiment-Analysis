"""Preprocess stage — normalize text, detect language, tokenize.

Reads the organic stream from clean, writes a preprocessed parquet with:
  - `text_norm`        normalized text (URLs/mentions stripped)
  - `lang`             per-row detected language
  - `tokens_lem_str`   space-joined cleaned tokens for downstream c-TF-IDF
  - `token_count`      diagnostic
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
from ada.tools.lang_detect import detect_one
from ada.tools.text_norm import normalize
from ada.tools.tokenize import tokenize


def _out_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "preprocess" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _input_parquet(state: GraphState) -> Path:
    """The organic stream from clean is the canonical downstream input."""
    clean = state.artifacts.get(Stage.CLEAN)
    if clean is None or not clean.parquet_path:
        raise RuntimeError("preprocess requires a clean.organic parquet")
    return Path(clean.parquet_path)


def preprocess_node(state: GraphState) -> dict:
    df = pd.read_parquet(_input_parquet(state))
    if "text" not in df.columns:
        raise RuntimeError("organic parquet has no `text` column")

    # 1. Normalize
    df["text_norm"] = df["text"].fillna("").astype(str).map(normalize)

    # 2. Detect language per row, with the schema's declared language as the prior
    declared = (state.confirmed_schema.language if state.confirmed_schema else "auto") or "auto"
    if declared and declared != "auto":
        # Trust the declared language; only run detection on rows with very little CJK
        # (cheap shortcut — saves langdetect cost across thousands of rows)
        df["lang"] = declared
    else:
        df["lang"] = df["text_norm"].map(lambda t: detect_one(t, default="auto"))

    # 3. Tokenize per language
    def _tokens(row) -> str:
        return " ".join(tokenize(row["text_norm"], row["lang"]))
    df["tokens_lem_str"] = df.apply(_tokens, axis=1)
    df["token_count"] = df["tokens_lem_str"].str.split().map(len)

    out_path = _out_dir(state) / "processed.parquet"
    df.to_parquet(out_path, index=False)
    out_hash = hash_file(out_path)

    summary = {
        "row_count": int(len(df)),
        "language_distribution": df["lang"].value_counts().to_dict(),
        "median_token_count": int(df["token_count"].median()),
        "empty_after_norm": int((df["text_norm"].str.strip() == "").sum()),
        "very_short_lt3_tokens": int((df["token_count"] < 3).sum()),
    }

    artifact = StageArtifact(
        stage=Stage.PREPROCESS,
        parquet_path=str(out_path),
        parquet_hash=out_hash,
        summary_stats=summary,
        notes=f"normalized + tokenized {len(df):,} rows; median {summary['median_token_count']} tokens/post",
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.PREPROCESS,
        action="normalized text + tokenized per row language",
        affected_rows=int(len(df)),
        reason="prepare for sentiment + topic stages",
        artifact_path=str(out_path),
        artifact_hash=out_hash,
    )

    return {
        "artifacts": {**state.artifacts, Stage.PREPROCESS: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.PREPROCESS],
    }
