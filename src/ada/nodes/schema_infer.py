"""Schema inference — LLM proposes a DatasetSchema; node queues a HITL question.

The planner sees the pending question and emits ASK_HUMAN. After the human
confirms, `confirmed_schema` is set and the planner moves on to reshape.

Falls back to a heuristic when the LLM is unavailable, mirroring the planner's
graceful-degradation pattern. Heuristic results are flagged in the audit so
the human knows to scrutinize them.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

import pandas as pd

from ada.llm.client import call_json
from ada.state import (
    AuditEntry,
    ColumnProfile,
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


_DATETIMELIKE = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2})?"
)
_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
_HAS_CJK = re.compile(r"[㐀-鿿]")
_TRADITIONAL_INDICATORS = re.compile(
    r"[颱這個們麼說發點時間實國灣]"
)


def _heuristic_schema(profiles: list[ColumnProfile]) -> dict:
    """Rule-based schema inference. Used when the LLM is unavailable."""
    by_name = {p.name: p for p in profiles}

    def looks_datetime(p: ColumnProfile) -> bool:
        return sum(1 for s in p.sample_values if _DATETIMELIKE.match(s.strip())) >= max(
            1, len(p.sample_values) // 2
        )

    def looks_numeric(p: ColumnProfile) -> bool:
        return sum(1 for s in p.sample_values if _NUMERIC.match(s.strip())) >= max(
            1, len(p.sample_values) // 2
        )

    def avg_len(p: ColumnProfile) -> float:
        if not p.sample_values:
            return 0.0
        return sum(len(s) for s in p.sample_values) / len(p.sample_values)

    # id_col: highest unique_pct among string-shaped columns
    id_candidates = sorted(
        [p for p in profiles if not looks_datetime(p) and not looks_numeric(p)],
        key=lambda p: -p.unique_pct,
    )
    id_col = id_candidates[0].name if id_candidates and id_candidates[0].unique_pct > 80 else None
    if id_col is None and profiles:
        id_col = max(profiles, key=lambda p: p.unique_pct).name

    # text_col: highest average sample length, excluding the id column
    text_candidates = [
        p for p in profiles
        if p.name != id_col and not looks_datetime(p) and not looks_numeric(p)
    ]
    text_col = max(text_candidates, key=avg_len).name if text_candidates else id_col

    # timestamp_col: any column whose samples parse as datetime
    timestamp_col = next((p.name for p in profiles if looks_datetime(p)), None)

    # engagement_col: numeric column
    engagement_col = next(
        (p.name for p in profiles if looks_numeric(p) and p.name not in {id_col}),
        None,
    )

    # platform_col / author_col: low-cardinality enums (≤ 20 unique, not text/id)
    enum_cols = [
        p for p in profiles
        if p.unique_pct <= 5  # rough enum threshold
        and p.name not in {id_col, text_col, timestamp_col, engagement_col}
        and not looks_numeric(p)
    ]
    # Heuristic: column name hints
    platform_col = None
    author_col = None
    for p in enum_cols:
        nm = p.name.lower()
        if platform_col is None and any(k in nm for k in ("platform", "source", "平台", "channel")):
            platform_col = p.name
        elif author_col is None and any(k in nm for k in ("author", "user", "帳號", "type", "role")):
            author_col = p.name
    # Fallback: first two enum cols
    remaining = [p.name for p in enum_cols if p.name not in {platform_col, author_col}]
    if platform_col is None and remaining:
        platform_col = remaining.pop(0)
    if author_col is None and remaining:
        author_col = remaining.pop(0)

    # Language: scan text samples
    text_samples = " ".join(by_name[text_col].sample_values) if text_col else ""
    if _HAS_CJK.search(text_samples):
        language = "zh-TW" if _TRADITIONAL_INDICATORS.search(text_samples) else "zh"
    else:
        language = "en"  # weak default

    return {
        "id_col": id_col,
        "text_col": text_col,
        "language": language,
        "timestamp_col": timestamp_col,
        "author_col": author_col,
        "engagement_col": engagement_col,
        "platform_col": platform_col,
        "extra_dims": {},
        "ambiguities": [],
        "confidence": "MODERATE",
        "needs_human": True,
        "needs_reshape": False,
        "reshape_hints": [],
        "_heuristic": True,
    }


def schema_infer_node(state: GraphState) -> dict:
    raw_count = state.artifacts[Stage.INGEST].summary_stats.get("row_count", 0)

    used_heuristic = False
    try:
        raw = call_json(
            "planner",
            "schema_inference",
            raw_file_path=str(state.raw_file_path),
            row_count=raw_count,
            user_initial_prompt=state.user_initial_prompt or "—",
            relevant_memory_excerpt=_memory_block(state),
        )
    except Exception as e:
        raw = _heuristic_schema(state.raw_columns)
        raw["_llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        used_heuristic = True

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

    source = "heuristic (LLM unavailable)" if used_heuristic else "LLM"
    artifact = StageArtifact(
        stage=Stage.SCHEMA_INFER,
        summary_stats={
            "source": source,
            "confidence": confidence,
            "needs_human": needs_human,
            "needs_reshape": needs_reshape,
            "ambiguity_count": len(ambiguities),
            **({"llm_error": raw["_llm_error"]} if used_heuristic else {}),
        },
        notes=f"{source} proposed schema; confidence={confidence}; "
              f"reshape_hints={len(reshape_hints)}",
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.SCHEMA_INFER,
        action=f"proposed schema via {source}",
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
