"""EDA stage — profile the canonical parquet, save figures, return summary stats.

Read-only with respect to data. Surfaces a HITL `open` question only if the
domain memory is thin AND the data shows interesting subgroup skew.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ada.config import settings
from ada.state import (
    AuditEntry,
    GraphState,
    HumanQuestion,
    QuestionType,
    Stage,
    StageArtifact,
)
from ada.tools import stats, viz


def _eda_dir(state: GraphState) -> Path:
    return settings.project_path(state.project_name) / "artifacts" / "eda" / state.run_id


def eda_node(state: GraphState) -> dict:
    if state.canonical_data_path is None:
        raise RuntimeError("eda requires a canonical parquet from reshape")

    df = pd.read_parquet(Path(state.canonical_data_path))
    out_dir = _eda_dir(state)

    summary = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "temporal": stats.temporal_summary(df),
        "platform": stats.categorical_summary(df, "platform"),
        "author": stats.categorical_summary(df, "author"),
        "engagement": stats.engagement_summary(df),
        "text_length": stats.text_length_summary(df),
        "quality": stats.quality_summary(df),
    }

    figures: list[str] = []
    for fn in (
        viz.temporal_chart,
        viz.platform_author_chart,
        viz.engagement_chart,
        viz.text_length_chart,
        viz.top_engagement_chart,
    ):
        try:
            p = fn(df, out_dir)
            if p is not None:
                figures.append(str(p))
        except Exception as e:  # noqa: BLE001 — chart failures should not abort EDA
            summary.setdefault("chart_errors", []).append(f"{fn.__name__}: {e!s:.120}")

    artifact = StageArtifact(
        stage=Stage.EDA,
        parquet_path=None,  # EDA doesn't transform data
        figure_paths=figures,
        summary_stats=summary,
        notes=f"profiled {len(df):,} rows; produced {len(figures)} figures",
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.EDA,
        action="profiled canonical data + generated figures",
        affected_rows=int(len(df)),
        reason="EDA pass before clean",
    )

    patch: dict = {
        "artifacts": {**state.artifacts, Stage.EDA: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.EDA],
    }

    # HITL trigger: dataset is heavily skewed AND we have no platform memory yet
    plat = summary["platform"]
    needs_open = (
        plat.get("available")
        and plat.get("top_pct")
        and max(plat["top_pct"].values()) > 60
        and not state.domain_knowledge.platforms
    )
    if needs_open:
        top_plat = max(plat["top_pct"].items(), key=lambda kv: kv[1])
        question = HumanQuestion(
            question_id=f"eda-skew-{uuid.uuid4().hex[:8]}",
            stage=Stage.EDA,
            question_type=QuestionType.OPEN,
            prompt=(
                f"This dataset is dominated by {top_plat[0]} ({top_plat[1]:.0f}%). "
                "What subgroup biases or known limitations of that platform should "
                "I record for downstream limitations?"
            ),
            payload={
                "platform_distribution": plat["top_pct"],
                "author_distribution": summary["author"].get("top_pct", {}),
                "time_window": {
                    "start": summary["temporal"].get("start"),
                    "end": summary["temporal"].get("end"),
                },
            },
            proposal=None,  # open question — no draft
            why_asking=f"top platform {top_plat[0]} > 60% of data and no platform memory",
            blocks_stage=False,  # informational; clean can proceed
        )
        patch["pending_questions"] = [*state.pending_questions, question]

    return patch
