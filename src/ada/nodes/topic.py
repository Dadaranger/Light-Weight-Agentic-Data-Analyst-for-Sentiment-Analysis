"""Topic stage — fit BERTopic, queue HITL question for analyst labels.

Day 3 of the course. Two corpora:
  - `text_norm`        for sentence embeddings (preserves context)
  - `tokens_lem_str`   for c-TF-IDF keyword extraction (already cleaned)

Auto-labels are generated from top keywords. The HITL question proposes them
and lets the human refine — this is the interrupt that actually changes
downstream brief quality the most, so it's `blocks_stage=True`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ada.config import settings
from ada.state import (
    AuditEntry,
    GraphState,
    HumanQuestion,
    QuestionType,
    Stage,
    StageArtifact,
)
from ada.tools.hashing import hash_file


MIN_TEXT_CHARS = 4         # filter very short normalized text out of the corpus
SAMPLES_PER_CLUSTER = 8    # examples shown to the human per topic


def _out_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "topic" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _input_parquet(state: GraphState) -> Path:
    sent = state.artifacts.get(Stage.SENTIMENT)
    if sent is None or not sent.parquet_path:
        raise RuntimeError("topic requires a sentiment-labeled parquet")
    return Path(sent.parquet_path)


def _representative_samples(df: pd.DataFrame, topic_id: int, n: int) -> list[dict]:
    """Pick top-engagement samples; fall back to first N if no engagement column."""
    sub = df[df["topic_id"] == topic_id]
    if sub.empty:
        return []
    if "engagement" in sub.columns:
        eng = pd.to_numeric(sub["engagement"], errors="coerce").fillna(0)
        sub = sub.assign(_eng=eng).nlargest(n, "_eng")
    else:
        sub = sub.head(n)
    return [
        {
            "id": str(r.get("id", "")),
            "platform": str(r.get("platform", "")),
            "text": str(r.get("text", ""))[:120],
            "sentiment": str(r.get("final_label", "")),
            "engagement": float(pd.to_numeric(r.get("engagement", 0), errors="coerce") or 0),
        }
        for _, r in sub.iterrows()
    ]


def topic_node(state: GraphState) -> dict:
    df = pd.read_parquet(_input_parquet(state))

    # Build the two corpora — drop rows where norm text is too short for clustering
    df["text_norm"] = df["text_norm"].fillna("").astype(str)
    df["tokens_lem_str"] = df["tokens_lem_str"].fillna("").astype(str)
    valid_mask = df["text_norm"].str.len() >= MIN_TEXT_CHARS
    df_valid = df.loc[valid_mask].reset_index(drop=True).copy()

    # Graceful degradation — too few rows to cluster. Mark stage done with a
    # diagnostic artifact instead of raising, so downstream stages can decide
    # whether to skip themselves or continue without topic info.
    if len(df_valid) < 30:
        skip_artifact = StageArtifact(
            stage=Stage.TOPIC,
            parquet_path=None,
            summary_stats={
                "skipped": True,
                "reason": f"only {len(df_valid)} valid posts; need ≥30 for clustering",
                "row_count": int(len(df_valid)),
            },
            notes=f"skipped — {len(df_valid)} rows < 30 minimum",
        )
        skip_audit = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            stage=Stage.TOPIC,
            action="skipped topic stage",
            affected_rows=int(len(df_valid)),
            reason=f"insufficient data ({len(df_valid)} rows < 30)",
        )
        return {
            "artifacts": {**state.artifacts, Stage.TOPIC: skip_artifact},
            "audit_log": [*state.audit_log, skip_audit],
            "completed_stages": [*state.completed_stages, Stage.TOPIC],
        }

    embed_corpus = df_valid["text_norm"].tolist()
    ctfidf_corpus = df_valid["tokens_lem_str"].tolist()

    # Pre-compute embeddings via our wrapper so the embedder is mockable in tests
    from ada.tools import embed, topic as topic_tools
    embeddings = embed.encode(embed_corpus)

    min_size = state.domain_knowledge.thresholds.min_topic_size
    language = (state.confirmed_schema.language if state.confirmed_schema else "chinese") or "chinese"
    bertopic_lang = "chinese" if language.startswith("zh") else "english"

    model, topic_assignments = topic_tools.fit_topics(
        embed_corpus=embed_corpus,
        ctfidf_corpus=ctfidf_corpus,
        embeddings=embeddings,
        min_topic_size=min_size,
        nr_topics="auto",
        language=bertopic_lang,
    )
    df_valid["topic_id"] = topic_assignments

    # Build per-cluster info — keywords, samples, draft label
    topic_ids = sorted({t for t in topic_assignments if t != -1})
    n_outliers = sum(1 for t in topic_assignments if t == -1)
    outlier_pct = round(n_outliers / max(len(df_valid), 1) * 100, 2)

    clusters: list[dict] = []
    auto_labels: dict[str, str] = {}
    for tid in topic_ids:
        keywords = topic_tools.top_keywords(model, tid, n=10)
        draft = topic_tools.auto_label(keywords, tid)
        auto_labels[str(tid)] = draft
        clusters.append({
            "topic_id": tid,
            "size": int((df_valid["topic_id"] == tid).sum()),
            "keywords": keywords,
            "draft_label": draft,
            "samples": _representative_samples(df_valid, tid, SAMPLES_PER_CLUSTER),
        })
    auto_labels["-1"] = "未歸類貼文（離群值）"

    # Apply draft labels — human may override via HITL
    df_valid["analyst_label"] = df_valid["topic_id"].astype(str).map(auto_labels)

    # Write preliminary parquet (will be rewritten by human handler if labels change)
    out_path = _out_dir(state) / "topics.parquet"
    df_valid.to_parquet(out_path, index=False)
    out_hash = hash_file(out_path)

    summary = {
        "row_count": int(len(df_valid)),
        "filtered_short_text": int((~valid_mask).sum()),
        "topic_count": len(topic_ids),
        "outlier_pct": outlier_pct,
        "topic_sizes": {str(tid): int((df_valid["topic_id"] == tid).sum()) for tid in topic_ids},
        "auto_labels": auto_labels,
        "min_topic_size": min_size,
    }
    if outlier_pct > state.domain_knowledge.thresholds.outlier_rerun_pct:
        summary["outlier_alarm"] = (
            f"outlier rate {outlier_pct}% exceeds threshold "
            f"{state.domain_knowledge.thresholds.outlier_rerun_pct}%"
        )

    artifact = StageArtifact(
        stage=Stage.TOPIC,
        parquet_path=str(out_path),
        parquet_hash=out_hash,
        summary_stats=summary,
        notes=(
            f"BERTopic found {len(topic_ids)} topics, "
            f"{n_outliers} outliers ({outlier_pct}%); awaiting HITL labels"
        ),
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.TOPIC,
        action="fit BERTopic + assigned topic_id per row",
        affected_rows=int(len(df_valid)),
        reason=f"topic_count={len(topic_ids)}, outlier_pct={outlier_pct}",
        artifact_path=str(out_path),
        artifact_hash=out_hash,
    )

    # HITL: one question with all clusters in the payload — human refines labels
    question = HumanQuestion(
        question_id=f"topic-label-{uuid.uuid4().hex[:8]}",
        stage=Stage.TOPIC,
        question_type=QuestionType.LABEL,
        prompt=(
            f"BERTopic found {len(topic_ids)} clusters. I drafted labels from the top "
            "keywords — please refine them based on the representative posts. "
            "Topic labels are the most consequential HITL decision in this pipeline; "
            "they propagate into the brief verbatim."
        ),
        payload={
            "clusters": clusters,
            "outlier_count": n_outliers,
            "outlier_pct": outlier_pct,
        },
        proposal={"labels": auto_labels, "approved": True},
        why_asking=f"BERTopic produced {len(topic_ids)} unlabeled clusters",
        blocks_stage=True,
    )

    return {
        "artifacts": {**state.artifacts, Stage.TOPIC: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.TOPIC],
        "pending_questions": [*state.pending_questions, question],
    }
