"""Slice 5 smoke test — narrative + amplification + brief produce a final brief file.

Reuses Slice 4's clustering setup (mocked embeddings) to land on a topic
parquet, then verifies the three downstream stages produce their artifacts and
the brief file contains the expected sections.
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
    """40 posts: 20 power-outage complaints + 20 community-help thanks."""
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
    rng = np.random.default_rng(42)
    n = len(texts); dim = 384
    centroid_a = rng.normal(0, 1, dim)
    centroid_b = rng.normal(5, 1, dim)
    out = np.zeros((n, dim), dtype="float32")
    for i, t in enumerate(texts):
        is_power = any(k in t for k in ["停電", "電力", "斷電", "斷水", "缺水"])
        c = centroid_a if is_power else centroid_b
        out[i] = (c + rng.normal(0, 0.3, dim)).astype("float32")
    return out


def test_slice5_full_pipeline_to_brief(medium_csv, project_dir, low_min_topic_size):
    from ada.graph import compile_graph
    from ada.memory.store import load_domain
    from ada.state import GraphState, Stage

    run_id = uuid4().hex[:12]
    state = GraphState(
        run_id=run_id,
        project_name="_test_slice5",
        started_at=datetime.now(timezone.utc),
        user_initial_prompt="taiwan typhoon test",
        raw_file_path=str(medium_csv.resolve()),
        domain_knowledge=load_domain("_test_slice5"),
    )
    graph = compile_graph()
    config = {"configurable": {"thread_id": run_id}}

    with patch("ada.nodes.schema_infer.call_json", side_effect=_mock_schema), \
         patch("ada.nodes.planner.call_json", side_effect=_mock_planner_unavailable), \
         patch("ada.tools.embed.encode", side_effect=_mock_encode):

        # Run to schema HITL
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass
        snapshot = graph.get_state(config)
        schema_q = next(t.interrupts[0].value for t in snapshot.tasks if t.interrupts)

        # Resume from schema → run through to topic HITL
        for _ in graph.stream(
            Command(resume=schema_q["proposal"]),
            config=config, stream_mode="updates",
        ):
            pass
        snapshot = graph.get_state(config)
        topic_q = next(
            (t.interrupts[0].value for t in snapshot.tasks if t.interrupts), None
        )
        assert topic_q is not None, "expected topic-label interrupt"

        # Resume from topic → should run narrative + amplification + brief to FINISH
        for _ in graph.stream(
            Command(resume=topic_q["proposal"]),
            config=config, stream_mode="updates",
        ):
            pass

    snapshot = graph.get_state(config)
    values = snapshot.values
    completed = [s.value if hasattr(s, "value") else s for s in values["completed_stages"]]
    for stage in ("narrative", "amplification", "brief"):
        assert stage in completed, f"{stage} not completed; got {completed}"

    artifacts = values["artifacts"]

    # ── Narrative ──────────────────────────────────────────────────────────
    narr = artifacts.get(Stage.NARRATIVE) or artifacts.get("narrative")
    assert narr is not None
    summary = narr.summary_stats if hasattr(narr, "summary_stats") else narr["summary_stats"]
    assert summary["narrative_count"] >= 2
    narratives_path = Path(summary["narratives_path"])
    assert narratives_path.exists()
    import json
    narratives = json.loads(narratives_path.read_text(encoding="utf-8"))
    # All narratives have the 6-element keys
    for n in narratives:
        for k in ("ACTOR", "ACTION", "VICTIM", "BLAME", "MORAL", "DESIRED", "NARRATIVE_STATEMENT"):
            assert k in n, f"narrative missing {k}"

    # ── Amplification ──────────────────────────────────────────────────────
    amp = artifacts.get(Stage.AMPLIFICATION) or artifacts.get("amplification")
    assert amp is not None
    asum = amp.summary_stats if hasattr(amp, "summary_stats") else amp["summary_stats"]
    assert asum["proxy_01_engagement"]["available"] is True
    assert asum["proxy_03_temporal"].get("available") in (True, False)  # may be False if too few hours
    assert asum["proxy_04_duplication"]["available"] is True
    coord = asum["coordination"]
    assert coord["coordination_confidence"] in ("HIGH", "MODERATE", "LOW")
    assert coord["total_count"] >= 4

    # ── Brief ──────────────────────────────────────────────────────────────
    brief = artifacts.get(Stage.BRIEF) or artifacts.get("brief")
    assert brief is not None
    bsum = brief.summary_stats if hasattr(brief, "summary_stats") else brief["summary_stats"]
    assert bsum["finding_count"] >= 1
    brief_path = Path(bsum["brief_path"])
    assert brief_path.exists()

    text = brief_path.read_text(encoding="utf-8")
    # Sanity-check the BLUF structure landed
    for header in (
        "Section 1 — Key findings",
        "Section 2 — Sentiment distribution",
        "Section 3 — Topic / narrative registry",
        "Section 4 — Amplification indicators",
        "Section 5 — Analytic limitations",
        "Section 6 — Recommendations",
    ):
        assert header in text, f"brief missing section header: {header}"
    assert "BLUF" in text
    assert "CONFIDENCE" in text
    assert "LIMITATION" in text

    # state.brief_path was set
    assert values.get("brief_path") == str(brief_path)
