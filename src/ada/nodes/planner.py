"""Planner node — decides what runs next.

Two-phase logic:
 1. Deterministic short-circuits handle obvious cases (pending question,
    nothing left to do, missing prerequisites). This keeps the LLM out of
    decisions that have only one right answer.
 2. Otherwise, build a state excerpt and call the planner LLM for a
    structured `PlannerDecision`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from ada.llm.client import call_json
from ada.state import (
    ExecutionPlan,
    GraphState,
    PlannerAction,
    PlannerDecision,
    Stage,
)

DEFAULT_PLAN: list[Stage] = [
    Stage.INGEST,
    Stage.SCHEMA_INFER,
    Stage.RESHAPE,
    Stage.EDA,
    Stage.CLEAN,
    Stage.PREPROCESS,
    Stage.SENTIMENT,
    Stage.TOPIC,
    Stage.NARRATIVE,
    Stage.AMPLIFICATION,
    Stage.BRIEF,
]


def _short_circuit(state: GraphState) -> PlannerDecision | None:
    """Handle decisions that don't require LLM judgment."""
    # Pending blocking question → resume it
    if state.has_blocking_questions():
        q = next(q for q in state.pending_questions if q.blocks_stage)
        return PlannerDecision(
            action=PlannerAction.ASK_HUMAN,
            question=q,
            reasoning=f"resuming pending question {q.question_id}",
        )

    # No plan yet → seed with default
    if state.plan is None:
        return PlannerDecision(
            action=PlannerAction.REPLAN,
            new_plan=DEFAULT_PLAN,
            reasoning="initial plan",
        )

    # Brief done → finish
    if Stage.BRIEF in state.completed_stages:
        return PlannerDecision(
            action=PlannerAction.FINISH,
            reasoning="brief written",
        )

    return None


def _next_stage(state: GraphState) -> Stage | None:
    if state.plan is None:
        return None
    for stage in state.plan.stages:
        if stage not in state.completed_stages:
            return stage
    return None


def _state_excerpt(state: GraphState) -> dict[str, str]:
    """Build the variable dict the planner prompt expects."""
    confirmed = (
        state.confirmed_schema.model_dump_json() if state.confirmed_schema else "—"
    )
    caps = (
        sorted(state.confirmed_schema.capabilities()) if state.confirmed_schema else []
    )
    plan = state.plan
    last_artifact = (
        state.artifacts.get(state.completed_stages[-1])
        if state.completed_stages
        else None
    )
    audit_tail = state.audit_log[-5:]
    recent_answers = state.answered_questions[-3:]
    return {
        "run_id": state.run_id,
        "project_name": state.project_name,
        "started_at": state.started_at.isoformat(),
        "user_initial_prompt": state.user_initial_prompt or "—",
        "confirmed_schema_json": confirmed,
        "capabilities_set": json.dumps(caps),
        "plan.revision": str(plan.revision if plan else 0),
        "plan.stages": json.dumps([s.value for s in (plan.stages if plan else [])]),
        "completed_stages": json.dumps([s.value for s in state.completed_stages]),
        "current_stage": state.current_stage.value if state.current_stage else "—",
        "domain_knowledge_excerpt": _memory_excerpt(state),
        "len(pending_questions)": str(len(state.pending_questions)),
        "list of question_ids": json.dumps([q.question_id for q in state.pending_questions]),
        "recent_answers": json.dumps(
            [
                {"q": q.question_id, "type": q.question_type.value, "ans": r.response}
                for q, r in recent_answers
            ],
            ensure_ascii=False,
        ),
        "completed_stages[-1]": (
            state.completed_stages[-1].value if state.completed_stages else "—"
        ),
        "path": str(last_artifact.parquet_path) if last_artifact else "—",
        "hash[:12]": (last_artifact.parquet_hash[:12] if last_artifact and last_artifact.parquet_hash else "—"),
        "summary_stats": json.dumps(last_artifact.summary_stats if last_artifact else {}),
        "notes": last_artifact.notes if last_artifact else "—",
        "audit_tail": json.dumps(
            [{"stage": e.stage.value, "action": e.action} for e in audit_tail],
            ensure_ascii=False,
        ),
    }


def _memory_excerpt(state: GraphState) -> str:
    dk = state.domain_knowledge
    parts: list[str] = [f"language={dk.language}"]
    if dk.platforms:
        parts.append(f"platforms_known={list(dk.platforms.keys())}")
    if dk.sarcasm_patterns:
        parts.append(f"sarcasm_patterns={len(dk.sarcasm_patterns)}")
    if dk.thresholds:
        parts.append(
            f"thresholds={{bot:{dk.thresholds.bot_quarantine_pct},"
            f"outlier:{dk.thresholds.outlier_rerun_pct}}}"
        )
    return " | ".join(parts)


def run_planner(state: GraphState) -> PlannerDecision:
    """Public entry — used by the LangGraph node wrapper."""
    sc = _short_circuit(state)
    if sc is not None:
        return sc

    # Try LLM. If it fails, fall back to "next stage in plan".
    try:
        raw = call_json("planner", "planner", **_state_excerpt(state))
        decision = PlannerDecision.model_validate(raw)
    except Exception as e:
        nxt = _next_stage(state)
        if nxt is None:
            return PlannerDecision(
                action=PlannerAction.FINISH,
                reasoning=f"planner LLM failed and no stages remain: {e}",
            )
        return PlannerDecision(
            action=PlannerAction.RUN_NODE,
            next_stage=nxt,
            reasoning=f"LLM unavailable, falling back to next planned stage: {e!s:.80}",
        )

    return decision


# ─────────────────────────────────────────────────────────────────────────────
# Node wrapper — applies REPLAN side effects
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: GraphState) -> dict:
    decision = run_planner(state)
    patch: dict = {"last_decision": decision}

    # REPLAN materializes the new plan into state.plan
    if decision.action == PlannerAction.REPLAN and decision.new_plan is not None:
        prev_rev = state.plan.revision if state.plan else 0
        patch["plan"] = ExecutionPlan(
            stages=decision.new_plan,
            created_at=datetime.now(timezone.utc),
            revision=prev_rev + 1,
            rationale=decision.reasoning,
        )

    # RUN_NODE updates current_stage so audit reflects what's executing
    if decision.action == PlannerAction.RUN_NODE and decision.next_stage:
        patch["current_stage"] = decision.next_stage

    return patch
