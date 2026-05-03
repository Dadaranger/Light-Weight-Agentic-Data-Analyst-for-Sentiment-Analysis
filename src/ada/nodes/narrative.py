"""Narrative stage — extract a 6-element narrative per topic.

Reads the topic-labeled parquet, generates one narrative per non-outlier topic
(LLM if available, template fallback if not). Output is a JSON sidecar that
the brief stage consumes; the parquet is unchanged.
"""
from __future__ import annotations

import json
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
from ada.tools.narrative import extract_narrative


SAMPLES_FOR_NARRATIVE = 8


def _out_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "narrative" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def narrative_node(state: GraphState) -> dict:
    topic_artifact = state.artifacts.get(Stage.TOPIC)
    if topic_artifact is None:
        raise RuntimeError("narrative requires the topic stage artifact")

    # Topic may have been skipped (small data) — narrative skips too
    if topic_artifact.summary_stats.get("skipped"):
        skip_artifact = StageArtifact(
            stage=Stage.NARRATIVE,
            summary_stats={
                "skipped": True,
                "reason": "topic stage was skipped — no clusters to narrate",
            },
            notes="skipped — upstream topic was skipped",
        )
        skip_audit = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            stage=Stage.NARRATIVE,
            action="skipped narrative stage",
            reason="upstream topic was skipped",
        )
        return {
            "artifacts": {**state.artifacts, Stage.NARRATIVE: skip_artifact},
            "audit_log": [*state.audit_log, skip_audit],
            "completed_stages": [*state.completed_stages, Stage.NARRATIVE],
        }

    if not topic_artifact.parquet_path:
        raise RuntimeError("narrative requires a topic parquet")

    df = pd.read_parquet(topic_artifact.parquet_path)
    if "topic_id" not in df.columns:
        raise RuntimeError("topic parquet missing topic_id column")

    language = (state.confirmed_schema.language if state.confirmed_schema else "zh-TW") or "zh-TW"
    auto_labels = topic_artifact.summary_stats.get("auto_labels", {})

    topic_ids = sorted(t for t in df["topic_id"].unique() if t != -1)
    narratives: list[dict] = []
    llm_errors = 0

    for tid in topic_ids:
        sub = df[df["topic_id"] == tid]
        label = (
            sub["analyst_label"].iloc[0]
            if "analyst_label" in sub.columns and not sub["analyst_label"].isna().all()
            else auto_labels.get(str(tid), f"T{tid}")
        )
        # Pick the highest-engagement samples for narrative grounding
        if "engagement" in sub.columns:
            eng = pd.to_numeric(sub["engagement"], errors="coerce").fillna(0)
            top = sub.assign(_e=eng).nlargest(SAMPLES_FOR_NARRATIVE, "_e")
        else:
            top = sub.head(SAMPLES_FOR_NARRATIVE)
        samples = top["text"].astype(str).tolist()

        # Pull keywords from topic artifact's auto_label (format: "Tnn：kw1·kw2·kw3（…）")
        auto_label_str = auto_labels.get(str(tid), "")
        keywords: list[str] = []
        if "：" in auto_label_str:
            kw_part = auto_label_str.split("：", 1)[1]
            kw_part = kw_part.split("（", 1)[0]
            keywords = [k for k in kw_part.split("·") if k]

        narrative = extract_narrative(label, keywords, samples, language)
        if narrative.get("_source") != "llm":
            llm_errors += 1
        narrative["topic_id"] = int(tid)
        narrative["analyst_label"] = label
        narrative["post_count"] = int(len(sub))
        narratives.append(narrative)

    # Persist as JSON sidecar (the parquet doesn't gain new columns)
    out_path = _out_dir(state) / "narratives.json"
    out_path.write_text(
        json.dumps(narratives, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "narrative_count": len(narratives),
        "narratives_path": str(out_path),
        "source_breakdown": {
            "llm": sum(1 for n in narratives if n.get("_source") == "llm"),
            "template": llm_errors,
        },
    }

    artifact = StageArtifact(
        stage=Stage.NARRATIVE,
        parquet_path=None,
        summary_stats=summary,
        notes=(
            f"extracted {len(narratives)} narratives "
            f"(LLM={summary['source_breakdown']['llm']}, "
            f"template={summary['source_breakdown']['template']})"
        ),
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.NARRATIVE,
        action="extracted 6-element narratives per topic",
        affected_rows=len(narratives),
        reason=f"{summary['source_breakdown']['llm']} via LLM, {llm_errors} via template",
    )

    return {
        "artifacts": {**state.artifacts, Stage.NARRATIVE: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.NARRATIVE],
    }
