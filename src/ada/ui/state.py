"""Session-state helpers for the Streamlit UI.

The graph + checkpointer live in `st.session_state` so they survive Streamlit
reruns. We keep an in-memory MemorySaver per session — runs are isolated to
a single browser session.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Suppress the LangGraph deserialization warnings when reading checkpoint state
warnings.filterwarnings("ignore", message="Deserializing unregistered type.*")

from langgraph.checkpoint.memory import MemorySaver

from ada.graph import _ada_serializer, compile_graph
from ada.memory.store import load_domain
from ada.state import GraphState, Stage


PAGE_START = "start"
PAGE_PROGRESS = "progress"
PAGE_HITL = "hitl"
PAGE_REPORT = "report"


def init_session() -> None:
    """Initialize session keys on first load."""
    ss = st.session_state
    if "checkpointer" not in ss:
        ss.checkpointer = MemorySaver(serde=_ada_serializer())
    if "graph" not in ss:
        ss.graph = compile_graph(checkpointer=ss.checkpointer)
    if "page" not in ss:
        ss.page = PAGE_START
    if "run_id" not in ss:
        ss.run_id = None
    if "project_name" not in ss:
        ss.project_name = ""
    if "language" not in ss:
        ss.language = "zh-TW"
    if "audit_seen" not in ss:
        ss.audit_seen = 0


def goto(page: str) -> None:
    st.session_state.page = page


def current_config() -> dict | None:
    if not st.session_state.run_id:
        return None
    return {"configurable": {"thread_id": st.session_state.run_id}}


def get_snapshot():
    cfg = current_config()
    if cfg is None:
        return None
    return st.session_state.graph.get_state(cfg)


def get_state_values() -> dict | None:
    snap = get_snapshot()
    return snap.values if snap and snap.values else None


def get_pending_interrupt() -> dict | None:
    """Return the first pending interrupt payload, or None."""
    snap = get_snapshot()
    if snap is None or not snap.next:
        return None
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def is_done() -> bool:
    """Run has finished — graph hit END and produced a brief."""
    values = get_state_values()
    if not values:
        return False
    completed = values.get("completed_stages", [])
    completed_values = {
        s.value if hasattr(s, "value") else s for s in completed
    }
    snap = get_snapshot()
    return Stage.BRIEF.value in completed_values and not (snap and snap.next)


def start_run(uploaded_file_path: Path, project: str, prompt: str) -> str:
    """Initialize a GraphState and run the pipeline until first interrupt or done."""
    run_id = uuid4().hex[:12]
    st.session_state.run_id = run_id
    st.session_state.project_name = project
    st.session_state.audit_seen = 0

    state = GraphState(
        run_id=run_id,
        project_name=project,
        started_at=datetime.now(timezone.utc),
        user_initial_prompt=prompt,
        raw_file_path=str(uploaded_file_path.resolve()),
        domain_knowledge=load_domain(project),
    )
    cfg = {"configurable": {"thread_id": run_id}}
    # Drain stream to first interrupt / END
    for _ in st.session_state.graph.stream(state, config=cfg, stream_mode="updates"):
        pass

    # Update language from confirmed schema as soon as it's set (it isn't yet
    # at this point — schema HITL hasn't fired). Default to zh-TW.
    return run_id


def resume_with(response: dict) -> None:
    """Pass a HITL response back into the graph and drain to next stop."""
    from langgraph.types import Command
    cfg = current_config()
    if cfg is None:
        return
    for _ in st.session_state.graph.stream(
        Command(resume=response), config=cfg, stream_mode="updates",
    ):
        pass

    # Refresh language from confirmed schema if available
    values = get_state_values() or {}
    schema = values.get("confirmed_schema")
    if schema is not None:
        lang = getattr(schema, "language", None) or (
            schema.get("language") if isinstance(schema, dict) else None
        )
        if lang:
            st.session_state.language = lang


def reset() -> None:
    """Forget the current run (does not delete artifacts on disk)."""
    st.session_state.run_id = None
    st.session_state.project_name = ""
    st.session_state.audit_seen = 0
    st.session_state.page = PAGE_START
