"""Reshape stage — produce canonical parquet from raw.

Slice 1 scope: rename columns to canonical names, parse timestamps if present,
cast id_col to string. Full ReshapeRecipe execution (concat, aggregate,
explode, melt) lands in a later slice.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ada.state import (
    AuditEntry,
    GraphState,
    Stage,
    StageArtifact,
)
from ada.tools.hashing import hash_file
from ada.tools.loader import load_dataset


CANONICAL_RENAMES = {
    "id_col": "id",
    "text_col": "text",
    "timestamp_col": "ts",
    "author_col": "author",
    "engagement_col": "engagement",
    "platform_col": "platform",
}


def _canonical_path(state: GraphState) -> "pd.io.common._PathLike":
    from ada.config import settings
    project = settings.project_path(state.project_name)
    out_dir = project / "artifacts" / "reshape"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{state.run_id}.parquet"


def reshape_node(state: GraphState) -> dict:
    from pathlib import Path
    schema = state.confirmed_schema
    if schema is None:
        raise RuntimeError("reshape requires confirmed_schema; planner should have waited for HITL")

    df = load_dataset(Path(state.raw_file_path))

    # Build rename map from confirmed schema
    rename_map: dict[str, str] = {}
    for schema_field, canonical in CANONICAL_RENAMES.items():
        src = getattr(schema, schema_field)
        if src and src in df.columns:
            rename_map[src] = canonical
    df = df.rename(columns=rename_map)

    # Cast id to string (always)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    # Parse timestamp if present
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    # Numeric coercion for engagement
    if "engagement" in df.columns:
        df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce")

    out_path = _canonical_path(state)
    df.to_parquet(out_path, index=False)
    canonical_hash = hash_file(out_path)

    artifact = StageArtifact(
        stage=Stage.RESHAPE,
        parquet_path=str(out_path),
        parquet_hash=canonical_hash,
        summary_stats={
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "renames_applied": rename_map,
            "ts_parsed": "ts" in df.columns,
            "ts_null_after_parse": int(df["ts"].isna().sum()) if "ts" in df.columns else None,
        },
        notes=f"renamed {len(rename_map)} columns to canonical names; wrote {out_path.name}",
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.RESHAPE,
        action="renamed and typed columns; wrote canonical parquet",
        affected_rows=int(len(df)),
        reason="schema confirmed; minimal reshape (Slice 1)",
        artifact_path=str(out_path),
        artifact_hash=canonical_hash,
    )

    return {
        "canonical_data_path": str(out_path),
        "canonical_hash": canonical_hash,
        "artifacts": {**state.artifacts, Stage.RESHAPE: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.RESHAPE],
    }
