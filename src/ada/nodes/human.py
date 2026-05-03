"""Human-in-the-loop response integration.

When the graph resumes after `interrupt()`, the response payload arrives here.
We dispatch on `(stage, question_type)` to apply the answer to the right
state field, then move the question from `pending_questions` to
`answered_questions`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

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


HANDLERS: dict[tuple[Stage, QuestionType], Handler] = {
    (Stage.SCHEMA_INFER, QuestionType.CONFIRM): _handle_schema_confirm,
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
        timestamp=datetime.utcnow(),
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
