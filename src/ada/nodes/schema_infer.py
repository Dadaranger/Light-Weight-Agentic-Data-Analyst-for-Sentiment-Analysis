"""Schema inference — LLM proposes a DatasetSchema; node queues a HITL question.

The planner sees the pending question and emits ASK_HUMAN. After the human
confirms, `confirmed_schema` is set and the planner moves on to reshape.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from ada.llm.client import call_json
from ada.state import (
    AuditEntry,
    DatasetSchema,
    GraphState,
    HumanQuestion,
    QuestionType,
    Stage,
    StageArtifact,
)


def _column_profile_block(state: GraphState) -> str:
    lines = []
    for c in state.raw_columns:
        lines.append(
            f"  - name: {c.name!r}\n"
            f"    dtype: {c.dtype}\n"
            f"    null_pct: {c.null_pct}\n"
            f"    unique_pct: {c.unique_pct}\n"
            f"    samples: {c.sample_values}"
        )
    return "\n".join(lines)


def _memory_block(state: GraphState) -> str:
    dk = state.domain_knowledge
    if not dk.platforms and not dk.notes:
        return "—"
    return json.dumps(
        {
            "language": dk.language,
            "platforms": list(dk.platforms.keys()),
            "notes": dk.notes,
        },
        ensure_ascii=False,
    )


def schema_infer_node(state: GraphState) -> dict:
    raw_count = state.artifacts[Stage.INGEST].summary_stats.get("row_count", 0)

    raw = call_json(
        "planner",
        "schema_inference",
        raw_file_path=str(state.raw_file_path),
        row_count=raw_count,
        user_initial_prompt=state.user_initial_prompt or "—",
        relevant_memory_excerpt=_memory_block(state),
    )

    # LLM returns either two sibling objects merged, or two top-level keys.
    # Be tolerant: allow flat or {"schema": ..., "meta": ...} shapes.
    if "error" in raw:
        raise RuntimeError(f"schema_inference failed: {raw['error']} — {raw.get('message', '')}")

    schema_fields = {
        k: raw.get(k)
        for k in (
            "id_col", "text_col", "language",
            "timestamp_col", "author_col", "engagement_col", "platform_col",
            "extra_dims",
        )
        if raw.get(k) is not None
    }
    proposed = DatasetSchema.model_validate(schema_fields)

    needs_human = bool(raw.get("needs_human", True))
    confidence = raw.get("confidence", "MODERATE")
    ambiguities = raw.get("ambiguities", [])
    needs_reshape = bool(raw.get("needs_reshape", False))
    reshape_hints = raw.get("reshape_hints", [])

    artifact = StageArtifact(
        stage=Stage.SCHEMA_INFER,
        summary_stats={
            "confidence": confidence,
            "needs_human": needs_human,
            "needs_reshape": needs_reshape,
            "ambiguity_count": len(ambiguities),
        },
        notes=f"LLM proposed schema; confidence={confidence}; "
              f"reshape_hints={len(reshape_hints)}",
    )

    audit = AuditEntry(
        timestamp=datetime.utcnow(),
        stage=Stage.SCHEMA_INFER,
        action="proposed schema",
        reason=f"confidence={confidence}, ambiguities={len(ambiguities)}",
    )

    patch: dict = {
        "proposed_schema": proposed,
        "artifacts": {**state.artifacts, Stage.SCHEMA_INFER: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.SCHEMA_INFER],
    }

    # Always confirm with human in Slice 1 — schema is high-stakes.
    # (Future: skip if confidence == HIGH and no ambiguities.)
    question = HumanQuestion(
        question_id=f"schema-{uuid.uuid4().hex[:8]}",
        stage=Stage.SCHEMA_INFER,
        question_type=QuestionType.CONFIRM,
        prompt=(
            "I propose this column → role mapping. Confirm or correct. "
            "If you reject, I'll re-infer with your hint."
        ),
        payload={
            "proposed_schema": proposed.model_dump(),
            "ambiguities": ambiguities,
            "reshape_hints": reshape_hints,
            "column_profiles": [c.model_dump() for c in state.raw_columns],
        },
        proposal={"schema": proposed.model_dump(), "approved": True},
        why_asking=(
            f"schema confidence={confidence}, "
            f"ambiguities={len(ambiguities)}, reshape_hints={len(reshape_hints)}"
        ),
        blocks_stage=True,
    )
    patch["pending_questions"] = [*state.pending_questions, question]
    return patch
