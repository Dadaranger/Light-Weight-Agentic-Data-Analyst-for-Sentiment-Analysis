"""Slice 3 smoke test — preprocess + sentiment produce labeled parquet."""
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
    """Mix of clearly positive, clearly negative, and ambiguous Chinese posts."""
    df = pd.DataFrame({
        "post_id": [f"p{i}" for i in range(10)],
        "post_text": [
            "感謝志工幫忙搬東西，真的很溫暖",       # POSITIVE
            "颱風來襲，請小心安全",                   # NEUTRAL
            "停電一週，物資短缺，太慘了",            # NEGATIVE
            "求救！我被困在三樓，水快淹進來了",      # NEGATIVE-DISTRESS
            "感謝政府的英明決策 👏",                  # UNCERTAIN (sarcasm)
            "希望大家都平安，加油！",                # POSITIVE
            "政府反應太慢，根本不可接受",            # NEGATIVE
            "已經撤離到安全地點，謝謝鄰居幫忙",      # POSITIVE
            "崩潰了，東西都被沖走，絕望",            # NEGATIVE-DISTRESS
            "在等政府公告",                           # NEUTRAL
        ],
        "created_at": ["2024-10-03 09:00:00"] * 10,
        "platform": ["PTT", "Dcard", "Facebook", "PTT", "PTT",
                     "Facebook", "PTT", "Dcard", "Facebook", "PTT"],
        "engagement": ["10", "5", "200", "1500", "300", "50", "800", "60", "120", "8"],
        "author_type": ["一般使用者"] * 10,
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


def _mock_schema(*args, **kwargs):
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


def test_slice3_preprocess_and_sentiment(tiny_csv, project_dir):
    from ada.graph import compile_graph
    from ada.memory.store import load_domain
    from ada.state import GraphState, Stage

    run_id = uuid4().hex[:12]
    state = GraphState(
        run_id=run_id,
        project_name="_test_slice3",
        started_at=datetime.now(timezone.utc),
        user_initial_prompt="taiwan typhoon test data",
        raw_file_path=str(tiny_csv.resolve()),
        domain_knowledge=load_domain("_test_slice3"),
    )
    graph = compile_graph()
    config = {"configurable": {"thread_id": run_id}}

    with patch("ada.nodes.schema_infer.call_json", side_effect=_mock_schema), \
         patch("ada.nodes.planner.call_json", side_effect=_mock_planner_unavailable):
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass
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
    completed = [s.value if hasattr(s, "value") else s for s in values["completed_stages"]]
    assert "preprocess" in completed
    assert "sentiment" in completed

    artifacts = values["artifacts"]
    pre = artifacts.get(Stage.PREPROCESS) or artifacts.get("preprocess")
    sent = artifacts.get(Stage.SENTIMENT) or artifacts.get("sentiment")
    assert pre is not None and sent is not None

    pre_path = pre.parquet_path if hasattr(pre, "parquet_path") else pre["parquet_path"]
    sent_path = sent.parquet_path if hasattr(sent, "parquet_path") else sent["parquet_path"]
    assert Path(pre_path).exists()
    assert Path(sent_path).exists()

    # ── Preprocess assertions ──────────────────────────────────────────────
    pre_df = pd.read_parquet(pre_path)
    assert "text_norm" in pre_df.columns
    assert "lang" in pre_df.columns
    assert "tokens_lem_str" in pre_df.columns
    assert (pre_df["lang"] == "zh-TW").all(), "schema declared zh-TW; should propagate"
    assert pre_df["tokens_lem_str"].str.len().gt(0).all(), "all rows should have tokens"

    # ── Sentiment assertions ───────────────────────────────────────────────
    sent_df = pd.read_parquet(sent_path)
    assert {"t1_label", "t2_label", "final_label", "sentiment_agreed"}.issubset(sent_df.columns)

    labels = set(sent_df["final_label"])
    # Should produce at least 3 distinct labels on this curated mini-corpus
    assert len(labels) >= 3, f"expected ≥3 distinct labels, got {labels}"

    # Sarcastic post should be UNCERTAIN
    sarcastic = sent_df[sent_df["text"].str.contains("感謝政府的英明", na=False)]
    assert len(sarcastic) == 1
    assert sarcastic.iloc[0]["final_label"] == "UNCERTAIN"

    # Distress posts should be flagged
    distress_terms = sent_df["text"].str.contains("被困|絕望", na=False)
    distress_labels = set(sent_df[distress_terms]["final_label"])
    assert "NEGATIVE-DISTRESS" in distress_labels

    # Audit covers all stages
    audit_stages = {
        e.stage.value if hasattr(e.stage, "value") else e.stage
        for e in values["audit_log"]
    }
    assert audit_stages >= {"ingest", "schema_infer", "reshape", "eda", "clean", "preprocess", "sentiment"}
