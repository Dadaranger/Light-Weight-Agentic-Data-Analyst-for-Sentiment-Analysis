"""Slice 4 smoke test — topic stage fits BERTopic, queues HITL labels,
human handler updates parquet with confirmed labels.

Embedder is mocked with random vectors clustered into two groups so BERTopic
finds 2 topics deterministically without downloading sentence-transformers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command


@pytest.fixture
def medium_csv(tmp_path: Path) -> Path:
    """40 posts split into two clear themes — power outage vs. volunteer help.
    Big enough for BERTopic with min_topic_size=5 to find 2 clusters.
    """
    power = [
        "停電好幾天了，電力公司根本沒在搶修",
        "我們社區停電，老人家很危險",
        "電力公司說要等到下週才能恢復供電",
        "停電中，冰箱的食物全壞了",
        "斷電兩天，孩子哭一整晚",
        "電力一直沒恢復，靠不住",
        "缺水又停電，物資也短缺",
        "停電的部分區域到底什麼時候能修好",
        "電力公司的搶修進度太慢了",
        "斷水斷電，連絡不到家人",
        "停電，孩子在家熱壞了",
        "電力公司延誤搶修，居民怒火",
        "停電狀況持續，希望快點修好",
        "電力中斷一週，已經受不了",
        "停電期間商家損失慘重",
        "電力供應不穩定，影響民生",
        "區域性停電造成不便",
        "電力恢復遙遙無期",
        "停電引發社區不滿",
        "電力故障的影響範圍擴大",
    ]
    helpers = [
        "感謝志工幫忙搬東西，真的很溫暖",
        "鄰居互相幫忙，台灣人的精神",
        "謝謝救難隊辛苦的付出",
        "志工團隊送來物資，真的很感動",
        "感謝社區的互助，大家一起度過難關",
        "鄰里之間互相支援，太棒了",
        "志工協助清理家園，真的很有愛心",
        "謝謝政府單位的快速應變",
        "感謝各方提供物資援助",
        "看到大家齊心協力，很感動",
        "志工幫忙照顧老人，真的很感謝",
        "謝謝醫護人員的辛苦",
        "感謝消防隊員的救援",
        "鄰居互助送餐，溫馨感人",
        "謝謝外地朋友送來物資",
        "感謝救助站的志工們",
        "互助合作讓災後復原更快",
        "志工的努力值得肯定",
        "感謝所有第一線人員",
        "鄰里互助是這次最大的收穫",
    ]
    rows = []
    for i, t in enumerate(power):
        rows.append({
            "post_id": f"power_{i}", "post_text": t,
            "created_at": "2024-10-03 10:00:00",
            "platform": "PTT", "engagement": str(50 + i),
            "author_type": "一般使用者",
        })
    for i, t in enumerate(helpers):
        rows.append({
            "post_id": f"help_{i}", "post_text": t,
            "created_at": "2024-10-04 10:00:00",
            "platform": "Facebook", "engagement": str(80 + i),
            "author_type": "一般使用者",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "medium.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch):
    proj_root = tmp_path / "projects"
    proj_root.mkdir()
    monkeypatch.setattr("ada.config.settings.projects_dir", proj_root)
    return proj_root


@pytest.fixture
def low_min_topic_size(monkeypatch):
    """Lower min_topic_size so BERTopic finds clusters in the small test corpus."""
    from ada.state import AnalysisThresholds
    original = AnalysisThresholds.__init__
    def patched(self, **kwargs):
        kwargs.setdefault("min_topic_size", 5)
        original(self, **kwargs)
    monkeypatch.setattr(AnalysisThresholds, "__init__", patched)


def _mock_schema(*args, **kwargs):
    return {
        "id_col": "post_id", "text_col": "post_text", "language": "zh-TW",
        "timestamp_col": "created_at", "platform_col": "platform",
        "engagement_col": "engagement", "author_col": "author_type",
        "extra_dims": {}, "ambiguities": [], "confidence": "HIGH",
        "needs_human": True, "needs_reshape": False, "reshape_hints": [],
    }


def _mock_planner_unavailable(*args, **kwargs):
    raise ConnectionError("ollama down")


def _mock_encode(texts, **kwargs):
    """Deterministic two-cluster embeddings — first half close to one centroid,
    second half close to another. Lets BERTopic find 2 distinct clusters.
    """
    rng = np.random.default_rng(42)
    n = len(texts)
    half = n // 2
    dim = 384
    centroid_a = rng.normal(0, 1, dim)
    centroid_b = rng.normal(5, 1, dim)
    out = np.zeros((n, dim), dtype="float32")
    for i, t in enumerate(texts):
        # Use the actual text content to decide cluster — texts about power vs help
        is_power = any(k in t for k in ["停電", "電力", "斷電", "斷水", "缺水"])
        c = centroid_a if is_power else centroid_b
        out[i] = (c + rng.normal(0, 0.3, dim)).astype("float32")
    return out


def test_slice4_topic_with_hitl(medium_csv, project_dir, low_min_topic_size):
    from ada.graph import compile_graph
    from ada.memory.store import load_domain
    from ada.state import GraphState, Stage

    run_id = uuid4().hex[:12]
    state = GraphState(
        run_id=run_id,
        project_name="_test_slice4",
        started_at=datetime.now(timezone.utc),
        user_initial_prompt="taiwan typhoon test",
        raw_file_path=str(medium_csv.resolve()),
        domain_knowledge=load_domain("_test_slice4"),
    )
    graph = compile_graph()
    config = {"configurable": {"thread_id": run_id}}

    with patch("ada.nodes.schema_infer.call_json", side_effect=_mock_schema), \
         patch("ada.nodes.planner.call_json", side_effect=_mock_planner_unavailable), \
         patch("ada.tools.embed.encode", side_effect=_mock_encode):

        # Up to schema HITL
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass
        snapshot = graph.get_state(config)
        schema_q = next(t.interrupts[0].value for t in snapshot.tasks if t.interrupts)
        assert schema_q["stage"] == "schema_infer"

        # Resume — should run through reshape, eda, clean, preprocess, sentiment, topic
        # then interrupt again at topic-label HITL
        for _ in graph.stream(
            Command(resume=schema_q["proposal"]),
            config=config, stream_mode="updates",
        ):
            pass

        snapshot = graph.get_state(config)
        topic_q = None
        for task in snapshot.tasks:
            if task.interrupts:
                topic_q = task.interrupts[0].value
                break
        assert topic_q is not None, "expected topic-label HITL interrupt"
        assert topic_q["stage"] == "topic"
        assert topic_q["type"] == "label"
        assert "clusters" in topic_q["payload"]
        assert len(topic_q["payload"]["clusters"]) >= 2, \
            f"expected ≥2 clusters, got {len(topic_q['payload']['clusters'])}"

        # Custom label override — emulate human refining the auto-labels
        custom_labels = dict(topic_q["proposal"]["labels"])  # start from auto
        first_tid = topic_q["payload"]["clusters"][0]["topic_id"]
        custom_labels[str(first_tid)] = "停電與電力供應問題"

        for _ in graph.stream(
            Command(resume={"labels": custom_labels, "approved": True}),
            config=config, stream_mode="updates",
        ):
            pass

    snapshot = graph.get_state(config)
    values = snapshot.values

    # Topic stage completed
    completed = [s.value if hasattr(s, "value") else s for s in values["completed_stages"]]
    assert "topic" in completed

    artifacts = values["artifacts"]
    topic_a = artifacts.get(Stage.TOPIC) or artifacts.get("topic")
    assert topic_a is not None
    parquet_path = topic_a.parquet_path if hasattr(topic_a, "parquet_path") else topic_a["parquet_path"]
    assert parquet_path and Path(parquet_path).exists()

    # Parquet has topic_id + analyst_label, with the human's override applied
    topic_df = pd.read_parquet(parquet_path)
    assert "topic_id" in topic_df.columns
    assert "analyst_label" in topic_df.columns

    # The custom label should have replaced the auto-label for that cluster
    matching = topic_df[topic_df["topic_id"] == first_tid]["analyst_label"].unique().tolist()
    assert "停電與電力供應問題" in matching, \
        f"expected custom label applied, got {matching}"

    # Summary stats record the confirmed labels
    summary = topic_a.summary_stats if hasattr(topic_a, "summary_stats") else topic_a["summary_stats"]
    assert "confirmed_labels" in summary

    # Audit log includes the topic stage
    audit_stages = {
        e.stage.value if hasattr(e.stage, "value") else e.stage
        for e in values["audit_log"]
    }
    assert "topic" in audit_stages
