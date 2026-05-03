"""Sentiment stage — Tier 1 rules + Tier 2 lexicon.

Reads preprocessed parquet, adds `t1_label`, `t2_label`, `final_label`, and a
`sentiment_agreed` flag. If T1/T2 disagreement exceeds the configured threshold,
queues a HITL `calibrate` question with 10 disputed examples.
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
from ada.tools.sentiment import combine, tier1_rules, tier2_lexicon


DISAGREEMENT_THRESHOLD_PCT = 15.0


def _out_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "sentiment" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _input_parquet(state: GraphState) -> Path:
    pre = state.artifacts.get(Stage.PREPROCESS)
    if pre is None or not pre.parquet_path:
        raise RuntimeError("sentiment requires a preprocess parquet")
    return Path(pre.parquet_path)


def _label_row(text_norm: str, tokens_str: str, language: str) -> tuple[str, float, str, float, str, bool]:
    t1, t1_conf = tier1_rules(text_norm, language)
    tokens = tokens_str.split() if isinstance(tokens_str, str) else []
    t2, t2_conf, _score = tier2_lexicon(tokens, language)
    final, agreed = combine(t1, t2, t1_conf, t2_conf)
    return t1, t1_conf, t2, t2_conf, final, agreed


def sentiment_node(state: GraphState) -> dict:
    df = pd.read_parquet(_input_parquet(state))
    needed = {"text_norm", "tokens_lem_str", "lang"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"sentiment needs columns from preprocess: missing {missing}")

    rows = df.apply(
        lambda r: _label_row(r["text_norm"], r["tokens_lem_str"], r["lang"]),
        axis=1, result_type="expand",
    )
    rows.columns = ["t1_label", "t1_conf", "t2_label", "t2_conf", "final_label", "sentiment_agreed"]
    df = pd.concat([df, rows], axis=1)

    out_path = _out_dir(state) / "labeled.parquet"
    df.to_parquet(out_path, index=False)
    out_hash = hash_file(out_path)

    disagreement_pct = round((~df["sentiment_agreed"]).sum() / max(len(df), 1) * 100, 2)
    label_dist = df["final_label"].value_counts().to_dict()
    summary = {
        "row_count": int(len(df)),
        "disagreement_pct": disagreement_pct,
        "label_distribution": label_dist,
        "uncertain_pct": round((df["final_label"] == "UNCERTAIN").sum() / max(len(df), 1) * 100, 2),
    }

    artifact = StageArtifact(
        stage=Stage.SENTIMENT,
        parquet_path=str(out_path),
        parquet_hash=out_hash,
        summary_stats=summary,
        notes=(
            f"labeled {len(df):,} rows; T1↔T2 disagreement {disagreement_pct}%; "
            f"distribution: {label_dist}"
        ),
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.SENTIMENT,
        action="applied Tier 1 (rules) + Tier 2 (lexicon) sentiment",
        affected_rows=int(len(df)),
        reason=f"baseline labeling; disagreement={disagreement_pct}%",
        artifact_path=str(out_path),
        artifact_hash=out_hash,
    )

    patch: dict = {
        "artifacts": {**state.artifacts, Stage.SENTIMENT: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.SENTIMENT],
    }

    # HITL trigger: T1/T2 disagree on > 15% of posts → calibrate
    if disagreement_pct > DISAGREEMENT_THRESHOLD_PCT:
        disputed = df[~df["sentiment_agreed"]].copy()
        # Pick the 10 highest-engagement disputed rows (most informative for calibration)
        if "engagement" in disputed.columns:
            disputed["_eng"] = pd.to_numeric(disputed["engagement"], errors="coerce").fillna(0)
            top = disputed.nlargest(10, "_eng")
        else:
            top = disputed.head(10)

        items = [
            {
                "id": str(row.get("id", "")),
                "text": str(row.get("text", ""))[:200],
                "t1_label": row["t1_label"],
                "t1_conf": float(row["t1_conf"]),
                "t2_label": row["t2_label"],
                "t2_conf": float(row["t2_conf"]),
                "final_label": row["final_label"],
            }
            for _, row in top.iterrows()
        ]

        question = HumanQuestion(
            question_id=f"sentiment-calibrate-{uuid.uuid4().hex[:8]}",
            stage=Stage.SENTIMENT,
            question_type=QuestionType.CALIBRATE,
            prompt=(
                f"My rule-based and lexicon-based labels disagree on {disagreement_pct}% of posts "
                f"(threshold {DISAGREEMENT_THRESHOLD_PCT}%). Please label these 10 high-impact "
                "examples and tell me what cultural/sarcasm patterns I'm missing."
            ),
            payload={
                "disagreement_pct": disagreement_pct,
                "items": items,
                "label_choices": ["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE-DISTRESS", "UNCERTAIN"],
            },
            proposal={
                "labels": [{"id": it["id"], "label": it["final_label"]} for it in items],
                "approved": True,
                "notes": "",
            },
            why_asking=f"T1/T2 disagreement {disagreement_pct}% > {DISAGREEMENT_THRESHOLD_PCT}% threshold",
            blocks_stage=False,  # don't block — calibration is a memory update, not a gate
        )
        patch["pending_questions"] = [*state.pending_questions, question]

    return patch
