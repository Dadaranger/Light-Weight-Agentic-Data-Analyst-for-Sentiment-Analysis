"""Amplification stage — compute four proxy indicators + coordination signals.

Pure deterministic computation. Reads the topic parquet (preferred) or falls
back to the sentiment parquet if topic was skipped. Bot share is read from
clean stage's summary stats.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ada.state import (
    AuditEntry,
    GraphState,
    Stage,
    StageArtifact,
)
from ada.tools.amplification import (
    coordination_signals,
    proxy_author_concentration,
    proxy_content_duplication,
    proxy_engagement_concentration,
    proxy_temporal_bursts,
)


def _select_input_parquet(state: GraphState) -> Path:
    # Prefer topic (has analyst_label for proxy 02), fall back to sentiment.
    topic = state.artifacts.get(Stage.TOPIC)
    if topic and topic.parquet_path:
        return Path(topic.parquet_path)
    sent = state.artifacts.get(Stage.SENTIMENT)
    if sent and sent.parquet_path:
        return Path(sent.parquet_path)
    raise RuntimeError("amplification requires topic or sentiment parquet")


def amplification_node(state: GraphState) -> dict:
    df = pd.read_parquet(_select_input_parquet(state))

    clean = state.artifacts.get(Stage.CLEAN)
    starting_rows = clean.summary_stats.get("starting_rows", 0) if clean else 0
    bot_rows = clean.summary_stats.get("bot_rows", 0) if clean else 0
    bot_share_pct = round(bot_rows / starting_rows * 100, 2) if starting_rows else 0.0

    th = state.domain_knowledge.thresholds

    p1 = proxy_engagement_concentration(df)
    p2 = proxy_author_concentration(df, topic_col="analyst_label" if "analyst_label" in df.columns else "platform")
    p3 = proxy_temporal_bursts(df, sigma=th.burst_sigma)
    p4 = proxy_content_duplication(df)

    coord = coordination_signals(
        p1, p2, p3, p4,
        bot_share_pct=bot_share_pct,
        duplication_alarm_pct=th.duplication_alarm_pct,
        bot_quarantine_pct=th.bot_quarantine_pct,
    )

    summary = {
        "proxy_01_engagement": p1,
        "proxy_02_author": p2,
        "proxy_03_temporal": p3,
        "proxy_04_duplication": p4,
        "bot_share_pct": bot_share_pct,
        "coordination": coord,
    }

    artifact = StageArtifact(
        stage=Stage.AMPLIFICATION,
        parquet_path=None,
        summary_stats=summary,
        notes=(
            f"coordination={coord['coordination_confidence']} "
            f"({coord['triggered_count']}/{coord['total_count']} signals); "
            f"bot share {bot_share_pct}%"
        ),
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.AMPLIFICATION,
        action="computed 4 amplification proxies + coordination checklist",
        affected_rows=int(len(df)),
        reason=(
            f"engagement_top5={p1.get('top5_share_pct', '—')}%, "
            f"duplication={p4.get('duplicate_pct', '—')}%, "
            f"bursts={p3.get('burst_count', '—')}"
        ),
    )

    return {
        "artifacts": {**state.artifacts, Stage.AMPLIFICATION: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.AMPLIFICATION],
    }
