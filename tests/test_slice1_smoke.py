"""End-to-end smoke test for Slice 1 (ingest → schema_infer → reshape).

LLM is mocked — this test verifies wiring, not LLM quality. Run against
a tiny synthetic CSV so no real data is needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pytest
from langgraph.types import Command


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "post_id": ["p1", "p2", "p3", "p4", "p5"],
        "post_text": [
            "颱風來襲，請小心",
            "停電了，物資不夠",
            "感謝志工幫忙搬東西",
            "政府反應太慢",
            "緊急轉傳：請所有人注意",
        ],
        "created_at": [
            "2024-10-01 09:00:00",
            "2024-10-02 14:30:00",
            "2024-10-03 08:15:00",
            "2024-10-03 18:00:00",
            "2024-10-04 06:45:00",
        ],
        "platform": ["PTT", "Dcard", "Facebook", "PTT", "LINE_社群"],
        "engagement": ["12", "5", "230", "1102", "44"],
    })
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch):
    """Point the projects dir at tmp so we don't touch the user's filesystem."""
    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    monkeypatch.setattr("ada.config.settings.projects_dir", proj_root)
    return proj_root


def _mock_schema_inference(*args, **kwargs) -> dict:
    """Return a deterministic schema proposal as the LLM would."""
    return {
        "id_col": "post_id",
        "text_col": "post_text",
        "language": "zh-TW",
        "timestamp_col": "created_at",
        "platform_col": "platform",
        "engagement_col": "engagement",
        "author_col": None,
        "extra_dims": {},
        "ambiguities": [],
        "confidence": "HIGH",
        "needs_human": True,
        "needs_reshape": False,
        "reshape_hints": [],
    }


def _mock_planner(*args, **kwargs) -> dict:
    """The planner short-circuits handle most of Slice 1 without touching the LLM.
    This mock fires only as a safety net — return a no-op REPLAN to default plan.
    """
    raise RuntimeError("planner LLM should not be called in Slice 1 happy path")


def test_slice1_end_to_end(tiny_csv, project_dir):
    """Run the graph all the way through ingest → schema → HITL → reshape.

    Asserts:
      - The graph interrupts after schema_infer for HITL confirmation
      - Auto-approving the proposal lets reshape complete
      - Canonical parquet exists, has the expected canonical column names
      - audit_log has entries for all three stages
    """
    from ada.graph import compile_graph
    from ada.memory.store import load_domain
    from ada.state import GraphState, Stage

    run_id = uuid4().hex[:12]
    state = GraphState(
        run_id=run_id,
        project_name="_test",
        started_at=datetime.now(timezone.utc),
        user_initial_prompt="taiwan typhoon test data",
        raw_file_path=str(tiny_csv.resolve()),
        domain_knowledge=load_domain("_test"),
    )

    graph = compile_graph()
    config = {"configurable": {"thread_id": run_id}}

    # Mock both LLM calls — schema_infer + planner fallback path
    with patch("ada.nodes.schema_infer.call_json", side_effect=_mock_schema_inference), \
         patch("ada.nodes.planner.call_json", side_effect=_mock_planner):

        # Initial run — should proceed through ingest, schema_infer, then interrupt
        for _ in graph.stream(state, config=config, stream_mode="values"):
            pass

        snapshot = graph.get_state(config)
        # Should be paused on human_gate
        assert snapshot.next, "graph should be paused after schema_infer"
        interrupt_value = None
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break
        assert interrupt_value is not None, "expected an interrupt payload"
        assert interrupt_value["stage"] == "schema_infer"
        assert interrupt_value["type"] == "confirm"
        assert "proposal" in interrupt_value
        assert interrupt_value["proposal"]["schema"]["text_col"] == "post_text"

        # Resume with the proposal
        resume_payload = interrupt_value["proposal"]
        for _ in graph.stream(Command(resume=resume_payload), config=config, stream_mode="values"):
            pass

    # Final state checks
    snapshot = graph.get_state(config)
    values = snapshot.values
    completed = values["completed_stages"]
    completed_values = [s.value if hasattr(s, "value") else s for s in completed]
    assert "ingest" in completed_values
    assert "schema_infer" in completed_values
    assert "reshape" in completed_values

    # Canonical parquet exists with expected columns
    canonical_path = values["canonical_data_path"]
    assert canonical_path is not None
    assert Path(str(canonical_path)).exists()
    df = pd.read_parquet(str(canonical_path))
    assert "id" in df.columns
    assert "text" in df.columns
    assert "ts" in df.columns
    assert "platform" in df.columns
    assert "engagement" in df.columns
    # Original column names should be gone
    assert "post_id" not in df.columns
    assert "post_text" not in df.columns
    # Types
    assert pd.api.types.is_datetime64_any_dtype(df["ts"]), "ts should be datetime"
    assert pd.api.types.is_string_dtype(df["id"])
    assert pd.api.types.is_numeric_dtype(df["engagement"])

    # Audit log spans all three stages
    audit_stages = {
        e.stage.value if hasattr(e.stage, "value") else e.stage
        for e in values["audit_log"]
    }
    assert audit_stages >= {"ingest", "schema_infer", "reshape"}

    # No leftover blocking questions
    pending = values["pending_questions"]
    assert all(not q.blocks_stage for q in pending), "should have no blocking questions left"
