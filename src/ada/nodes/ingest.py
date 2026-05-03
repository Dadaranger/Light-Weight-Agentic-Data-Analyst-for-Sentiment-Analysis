"""Ingest stage — hash the file, load it, profile columns. No transformations.

Outputs:
- raw_file_hash
- raw_columns: list[ColumnProfile]
- audit entry recording the load
- artifact: stage summary stats only (no parquet — that's reshape's job)
"""
from __future__ import annotations

from datetime import datetime, timezone

from ada.state import (
    AuditEntry,
    GraphState,
    Stage,
    StageArtifact,
)
from ada.tools.hashing import hash_file
from ada.tools.loader import load_dataset
from ada.tools.profile import profile_columns


def ingest_node(state: GraphState) -> dict:
    raw_path = state.raw_file_path
    file_hash = hash_file(raw_path)
    df = load_dataset(raw_path)
    profiles = profile_columns(df)

    artifact = StageArtifact(
        stage=Stage.INGEST,
        parquet_path=None,  # raw stays raw
        parquet_hash=None,
        figure_paths=[],
        summary_stats={
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "file_size_bytes": raw_path.stat().st_size,
            "encoding_used": "auto",
        },
        notes=f"loaded {raw_path.name}; profiled {len(df.columns)} columns",
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.INGEST,
        action="loaded raw file",
        affected_rows=int(len(df)),
        reason="initial ingest",
        artifact_path=raw_path,
        artifact_hash=file_hash,
    )

    return {
        "raw_file_hash": file_hash,
        "raw_columns": profiles,
        "artifacts": {**state.artifacts, Stage.INGEST: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.INGEST],
    }
