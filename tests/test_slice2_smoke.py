"""Slice 2 smoke test — graph runs through eda + clean and produces three streams."""
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
    """Includes one duplicate (p2 twice), one empty text (p4), and one bot (p5)."""
    df = pd.DataFrame({
        "post_id": ["p1", "p2", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
        "post_text": [
            "颱風來襲",
            "停電了",
            "停電了 (重複)",  # composite-key dup if same platform — see below
            "感謝志工",
            "",  # empty
            "緊急轉傳",
            "希望大家平安",
            "求助",
            "已撤離安全地點",
        ],
        "created_at": [
            "2024-10-01 09:00:00",
            "2024-10-02 14:30:00",
            "2024-10-02 14:31:00",
            "2024-10-03 08:15:00",
            "2024-10-03 18:00:00",
            "2024-10-04 06:45:00",
            "2024-10-04 12:00:00",
            "2024-10-05 09:00:00",
            "2024-10-06 11:00:00",
        ],
        # p2 + Dcard appears twice → composite-key dup (drops one)
        "platform": ["PTT", "Dcard", "Dcard", "Facebook", "PTT", "LINE_社群", "PTT", "Facebook", "PTT"],
        "engagement": ["12", "5", "5", "230", "1102", "44", "8", "100", "60"],
        "author_type": [
            "一般使用者", "一般使用者", "一般使用者", "意見領袖",
            "一般使用者", "疑似機器人", "一般使用者", "一般使用者", "一般使用者",
        ],
    })
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch):
    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    monkeypatch.setattr("ada.config.settings.projects_dir", proj_root)
    return proj_root


def _mock_schema_inference(*args, **kwargs) -> dict:
    return {
        "id_col": "post_id",
        "text_col": "post_text",
        "language": "zh-TW",
        "timestamp_col": "created_at",
        "platform_col": "platform",
        "engagement_col": "engagement",
        "author_col": "author_type",
        "extra_dims": {},
        "ambiguities": [],
        "confidence": "HIGH",
        "needs_human": True,
        "needs_reshape": False,
        "reshape_hints": [],
    }


def _mock_planner_unavailable(*args, **kwargs):
    raise ConnectionError("ollama down")


def test_slice2_eda_and_clean(tiny_csv, project_dir):
    """Run all the way through eda + clean. Three streams should land on disk."""
    from ada.graph import compile_graph
    from ada.memory.store import load_domain
    from ada.state import GraphState, Stage

    run_id = uuid4().hex[:12]
    state = GraphState(
        run_id=run_id,
        project_name="_test_slice2",
        started_at=datetime.now(timezone.utc),
        user_initial_prompt="taiwan typhoon test data",
        raw_file_path=str(tiny_csv.resolve()),
        domain_knowledge=load_domain("_test_slice2"),
    )

    graph = compile_graph()
    config = {"configurable": {"thread_id": run_id}}

    with patch("ada.nodes.schema_infer.call_json", side_effect=_mock_schema_inference), \
         patch("ada.nodes.planner.call_json", side_effect=_mock_planner_unavailable):
        # First pass — interrupts at schema HITL
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        # Resume — auto-approve schema, then graph continues through reshape, eda, clean
        snapshot = graph.get_state(config)
        interrupt_value = next(
            t.interrupts[0].value for t in snapshot.tasks if t.interrupts
        )
        for _ in graph.stream(
            Command(resume=interrupt_value["proposal"]),
            config=config, stream_mode="updates",
        ):
            pass

    snapshot = graph.get_state(config)
    values = snapshot.values

    completed_values = [
        s.value if hasattr(s, "value") else s
        for s in values["completed_stages"]
    ]
    assert "eda" in completed_values, f"eda missing — got {completed_values}"
    assert "clean" in completed_values, f"clean missing — got {completed_values}"

    # ── EDA artifact assertions ────────────────────────────────────────────
    artifacts = values["artifacts"]
    eda = artifacts.get(Stage.EDA) or artifacts.get("eda")
    assert eda is not None
    summary = eda.summary_stats if hasattr(eda, "summary_stats") else eda["summary_stats"]
    assert summary["row_count"] == 9
    assert summary["temporal"]["available"] is True
    assert summary["platform"]["available"] is True
    assert summary["author"]["available"] is True
    assert summary["engagement"]["available"] is True
    assert summary["quality"]["composite_dup_count"] == 1, "expected one composite dup"
    assert summary["quality"]["empty_text_count"] == 1, "expected one empty text"

    figs = eda.figure_paths if hasattr(eda, "figure_paths") else eda["figure_paths"]
    assert len(figs) >= 3, f"expected ≥3 figures, got {len(figs)}"
    for f in figs:
        assert Path(f).exists(), f"figure missing: {f}"

    # ── CLEAN artifact assertions ──────────────────────────────────────────
    clean = artifacts.get(Stage.CLEAN) or artifacts.get("clean")
    assert clean is not None
    csum = clean.summary_stats if hasattr(clean, "summary_stats") else clean["summary_stats"]

    # Starting 9, dedup removes 1 → 8, quarantine removes 1 (empty) → 7, bot split removes 1 → 6 organic
    assert csum["starting_rows"] == 9
    assert csum["organic_rows"] == 6
    assert csum["quarantine_rows"] == 1
    assert csum["bot_rows"] == 1

    # All three parquets exist
    streams = csum["streams"]
    assert Path(streams["organic"]).exists()
    assert Path(streams["quarantine"]).exists()
    assert Path(streams["bots"]).exists()

    # Organic has the right shape
    organic = pd.read_parquet(streams["organic"])
    assert len(organic) == 6
    assert "疑似機器人" not in organic["author"].values
    assert organic["text"].str.strip().ne("").all()

    # Bots stream contains the bot
    bots = pd.read_parquet(streams["bots"])
    assert len(bots) == 1
    assert (bots["author"] == "疑似機器人").all()

    # Audit log has entries from all five stages
    audit_stages = {
        e.stage.value if hasattr(e.stage, "value") else e.stage
        for e in values["audit_log"]
    }
    assert audit_stages >= {"ingest", "schema_infer", "reshape", "eda", "clean"}
