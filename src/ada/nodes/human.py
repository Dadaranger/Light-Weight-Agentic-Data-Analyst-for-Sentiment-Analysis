"""Human-in-the-loop response integration.

When the graph resumes after `interrupt()`, the response payload arrives here.
We dispatch on `(stage, question_type)` to apply the answer to the right
state field, then move the question from `pending_questions` to
`answered_questions`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ada.state import (
    DatasetSchema,
    GraphState,
    HumanQuestion,
    HumanResponse,
    QuestionType,
    Stage,
)

Handler = Callable[[GraphState, HumanQuestion, dict[str, Any]], dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Per-question-type handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_schema_confirm(state: GraphState, q: HumanQuestion, resp: dict) -> dict:
    """Response shape: { "schema": <DatasetSchema fields>, "approved": bool }.
    If `approved` is true and `schema` omitted, accept the proposal verbatim.
    """
    if resp.get("approved", True):
        schema_data = resp.get("schema") or (q.proposal or {}).get("schema")
        if schema_data is None:
            raise ValueError("schema_confirm: no schema in response or proposal")
        return {"confirmed_schema": DatasetSchema.model_validate(schema_data)}
    # rejected → leave confirmed_schema None; planner will replan or re-ask
    return {}


def _handle_topic_label(state: GraphState, q: HumanQuestion, resp: dict) -> dict:
    """Response shape: { "labels": {topic_id_str: label_str}, "approved": bool }.
    Updates the analyst_label column in the topic parquet so downstream stages
    (narrative, brief) read the human-confirmed labels.
    """
    if not resp.get("approved", True):
        return {}
    labels = resp.get("labels") or (q.proposal or {}).get("labels") or {}
    if not labels:
        return {}

    topic_artifact = state.artifacts.get(Stage.TOPIC)
    if topic_artifact is None or not topic_artifact.parquet_path:
        return {}
    parquet_path = Path(topic_artifact.parquet_path)
    df = pd.read_parquet(parquet_path)
    df["analyst_label"] = (
        df["topic_id"].astype(str).map(labels).fillna(df.get("analyst_label"))
    )
    df.to_parquet(parquet_path, index=False)

    # Refresh artifact's summary so the audit reflects confirmed labels
    new_summary = {**topic_artifact.summary_stats, "confirmed_labels": labels}
    new_artifact = topic_artifact.model_copy(update={"summary_stats": new_summary})
    return {"artifacts": {**state.artifacts, Stage.TOPIC: new_artifact}}


HANDLERS: dict[tuple[Stage, QuestionType], Handler] = {
    (Stage.SCHEMA_INFER, QuestionType.CONFIRM): _handle_schema_confirm,
    (Stage.TOPIC, QuestionType.LABEL): _handle_topic_label,
}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def integrate_response(state: GraphState, question: HumanQuestion, resp: Any) -> dict:
    if not isinstance(resp, dict):
        raise TypeError(f"human response must be a dict, got {type(resp).__name__}")

    patch: dict = {}

    handler = HANDLERS.get((question.stage, question.question_type))
    if handler is not None:
        patch.update(handler(state, question, resp))

    # Always drain the question from pending → answered
    response_obj = HumanResponse(
        question_id=question.question_id,
        response=resp,
        timestamp=datetime.now(timezone.utc),
        persist_to_memory=bool(resp.get("persist_to_memory", False)),
    )
    patch["pending_questions"] = [
        q for q in state.pending_questions if q.question_id != question.question_id
    ]
    patch["answered_questions"] = [
        *state.answered_questions,
        (question, response_obj),
    ]
    return patch
