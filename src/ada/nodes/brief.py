"""Brief stage — produce the final BLUF-format integrated analytic brief.

Reads narrative + amplification + topic + sentiment artifacts, picks 2-3
findings, and renders a text file in the language declared by
`confirmed_schema.language` (zh-TW or en). Template-based — LLM is not
required for the rendering itself, only for narrative extraction upstream.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ada.config import settings
from ada.i18n import t_brief
from ada.state import (
    AuditEntry,
    BLUFFinding,
    Confidence,
    GraphState,
    Stage,
    StageArtifact,
)


def _out_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "brief" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _topic_sentiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if "analyst_label" not in df.columns or "final_label" not in df.columns:
        return pd.DataFrame()
    return (
        pd.crosstab(df["analyst_label"], df["final_label"], normalize="index").mul(100).round(1)
    )


def _negative_share(matrix: pd.DataFrame) -> pd.Series:
    neg_cols = [c for c in ("NEGATIVE", "NEGATIVE-DISTRESS") if c in matrix.columns]
    if not neg_cols:
        return pd.Series(dtype=float)
    return matrix[neg_cols].sum(axis=1).sort_values(ascending=False)


def _positive_share(matrix: pd.DataFrame) -> pd.Series:
    if "POSITIVE" not in matrix.columns:
        return pd.Series(dtype=float)
    return matrix["POSITIVE"].sort_values(ascending=False)


def _find_narrative(narratives: list[dict], label: str) -> dict | None:
    for n in narratives:
        if n.get("analyst_label") == label:
            return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Findings
# ─────────────────────────────────────────────────────────────────────────────

def _build_findings(
    matrix: pd.DataFrame,
    narratives: list[dict],
    amp_summary: dict,
    language: str,
) -> list[BLUFFinding]:
    findings: list[BLUFFinding] = []
    coord = amp_summary.get("coordination", {})
    p1 = amp_summary.get("proxy_01_engagement", {})
    p4 = amp_summary.get("proxy_04_duplication", {})
    no_actor = t_brief(language, "no_actor")
    no_narr = t_brief(language, "narrative_unavailable")

    # 1 — highest-risk narrative
    neg = _negative_share(matrix)
    if not neg.empty and neg.iloc[0] > 0:
        topic = neg.index[0]
        pct = float(neg.iloc[0])
        narr = _find_narrative(narratives, topic) or {}
        narr_stmt = narr.get("NARRATIVE_STATEMENT", no_narr)
        actor = narr.get("ACTOR", no_actor)
        post_count = int(narr.get("post_count", 0))
        findings.append(BLUFFinding(
            title=t_brief(language, "finding_title_neg", topic=topic),
            bluf=t_brief(language, "bluf_neg_template", topic=topic, posts=post_count, pct=pct),
            evidence=t_brief(language, "evidence_neg_template", pct=pct, stmt=narr_stmt[:200]),
            confidence=Confidence.MODERATE,
            confidence_reason=t_brief(language, "conf_reason_neg"),
            limitations=[
                t_brief(language, "lim_neg_1"),
                t_brief(language, "lim_neg_2"),
            ],
            recommendation=t_brief(language, "rec_neg_template", topic=topic, actor=actor),
            related_topics=[topic],
        ))

    # 2 — highest positive (counter-narrative)
    pos = _positive_share(matrix)
    if not pos.empty and pos.iloc[0] > 0:
        topic = pos.index[0]
        pct = float(pos.iloc[0])
        narr = _find_narrative(narratives, topic) or {}
        narr_stmt = narr.get("NARRATIVE_STATEMENT", no_narr)
        post_count = int(narr.get("post_count", 0))
        findings.append(BLUFFinding(
            title=t_brief(language, "finding_title_pos", topic=topic),
            bluf=t_brief(language, "bluf_pos_template", topic=topic, posts=post_count, pct=pct),
            evidence=t_brief(language, "evidence_pos_template", pct=pct, stmt=narr_stmt[:200]),
            confidence=Confidence.MODERATE,
            confidence_reason=t_brief(language, "conf_reason_pos"),
            limitations=[
                t_brief(language, "lim_pos_1"),
                t_brief(language, "lim_pos_2"),
            ],
            recommendation=t_brief(language, "rec_pos_template", topic=topic),
            related_topics=[topic],
        ))

    # 3 — amplification
    triggered = coord.get("triggered_count", 0)
    if triggered >= 1:
        conf_str = coord.get("coordination_confidence", "LOW")
        conf_enum = {"HIGH": Confidence.HIGH, "MODERATE": Confidence.MODERATE, "LOW": Confidence.LOW}[conf_str]
        conf_localized = t_brief(language, f"confidence_{conf_str}")
        signals_summary = "; ".join(
            f"{s['name']} {'✓' if s['triggered'] else '✗'}"
            for s in coord.get("signals", [])
            if "not available" not in s["evidence"]
        )
        findings.append(BLUFFinding(
            title=t_brief(language, "finding_title_amp", conf=conf_localized),
            bluf=t_brief(
                language, "bluf_amp_template",
                triggered=triggered, total=coord.get("total_count", 0),
                top5=p1.get("top5_share_pct", "?"), dup=p4.get("duplicate_pct", "?"),
            ),
            evidence=t_brief(language, "evidence_amp_template", summary=signals_summary),
            confidence=conf_enum,
            confidence_reason=coord.get("confidence_reason", ""),
            limitations=[
                t_brief(language, "lim_amp_1"),
                t_brief(language, "lim_amp_2"),
            ],
            recommendation=t_brief(language, "rec_amp_template"),
            related_topics=[],
        ))

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def _render_brief(
    state: GraphState,
    findings: list[BLUFFinding],
    matrix: pd.DataFrame,
    narratives: list[dict],
    amp_summary: dict,
    df: pd.DataFrame,
    language: str,
) -> str:
    L: list[str] = []
    BAR = "━" * 70
    project = state.project_name
    today = date.today().isoformat()
    n_rows = len(df)
    schema = state.confirmed_schema
    caps = sorted(schema.capabilities()) if schema else []
    coord = amp_summary.get("coordination", {})
    p1 = amp_summary.get("proxy_01_engagement", {})
    p2 = amp_summary.get("proxy_02_author", {})
    p3 = amp_summary.get("proxy_03_temporal", {})
    p4 = amp_summary.get("proxy_04_duplication", {})

    title = t_brief(language, "title_template", project=project)
    L.append("╔" + "═" * 70 + "╗")
    L.append(f"║{('  ' + title).ljust(70)}║")
    L.append("╚" + "═" * 70 + "╝")
    L.append("")
    L.append(f"{t_brief(language, 'field_run_id')}:    {state.run_id}")
    L.append(f"{t_brief(language, 'field_project')}:   {project}")
    L.append(f"{t_brief(language, 'field_date')}:      {today}")
    L.append(
        f"{t_brief(language, 'field_dataset')}:   {Path(state.raw_file_path).name}, "
        f"{n_rows:,} {t_brief(language, 'field_rows')}"
    )
    L.append(f"{t_brief(language, 'field_capabilities')}: {caps}")
    L.append(f"{t_brief(language, 'field_status')}:    {t_brief(language, 'status_draft')}")
    L.append("")

    # Section 1 — findings
    L.append(BAR)
    L.append(t_brief(language, "section_1"))
    L.append(BAR)
    if not findings:
        L.append("  " + t_brief(language, "no_findings"))
    for i, f in enumerate(findings, 1):
        L.append("")
        L.append(f"[{i}] {f.title}")
        L.append(f"    {t_brief(language, 'label_bluf')}: {f.bluf}")
        L.append(f"    {t_brief(language, 'label_evidence')}: {f.evidence}")
        conf_localized = t_brief(language, f"confidence_{f.confidence.value}")
        L.append(f"    {t_brief(language, 'label_confidence')}: {conf_localized} — {f.confidence_reason}")
        for j, lim in enumerate(f.limitations, 1):
            L.append(f"    {t_brief(language, 'label_limitation')} {j}: {lim}")
        L.append(f"    {t_brief(language, 'label_recommend')}: {f.recommendation}")

    # Section 2 — sentiment matrix
    L.append("")
    L.append(BAR)
    L.append(t_brief(language, "section_2"))
    L.append(BAR)
    L.append("")
    if matrix.empty:
        L.append("  " + t_brief(language, "matrix_unavailable"))
    else:
        pos_lbl = t_brief(language, "matrix_pos")
        neg_lbl = t_brief(language, "matrix_neg")
        dst_lbl = t_brief(language, "matrix_dst")
        ntr_lbl = t_brief(language, "matrix_ntr")
        n_lbl = t_brief(language, "matrix_count")
        for topic in matrix.head(20).index:
            row = matrix.loc[topic]
            n_topic = int((df["analyst_label"] == topic).sum()) if "analyst_label" in df.columns else 0
            L.append(
                f"  {str(topic)[:32]:<32} | {pos_lbl} {row.get('POSITIVE', 0.0):5.1f}% | "
                f"{neg_lbl} {row.get('NEGATIVE', 0.0):5.1f}% | "
                f"{dst_lbl} {row.get('NEGATIVE-DISTRESS', 0.0):5.1f}% | "
                f"{ntr_lbl} {row.get('NEUTRAL', 0.0):5.1f}% | {n_lbl}={n_topic:,}"
            )
        L.append("")
        L.append("  " + t_brief(language, "matrix_warning_1"))
        L.append("  " + t_brief(language, "matrix_warning_2"))

    # Section 3 — narratives
    L.append("")
    L.append(BAR)
    L.append(t_brief(language, "section_3"))
    L.append(BAR)
    L.append("")
    no_actor = t_brief(language, "no_actor")
    for n in narratives[:20]:
        topic_lbl = t_brief(language, "narrative_topic")
        posts_lbl = t_brief(language, "narrative_posts")
        stmt_lbl = t_brief(language, "narrative_statement")
        actor_lbl = t_brief(language, "narrative_actor")
        desired_lbl = t_brief(language, "narrative_desired")
        L.append(f"  {topic_lbl}: {n.get('analyst_label', '—')}  ({n.get('post_count', 0):,} {posts_lbl})")
        L.append(f"  {stmt_lbl}: {n.get('NARRATIVE_STATEMENT', '—')[:240]}")
        L.append(
            f"  {actor_lbl}: {n.get('ACTOR', no_actor)}  |  "
            f"{desired_lbl}: {n.get('DESIRED', no_actor)}"
        )
        L.append("")

    # Section 4 — amplification
    L.append(BAR)
    L.append(t_brief(language, "section_4"))
    L.append(BAR)
    L.append("")
    L.append("  " + t_brief(language, "amp_proxy01", top5=p1.get("top5_share_pct", "—")))
    L.append("  " + t_brief(language, "amp_proxy02", n_topics=len(p2.get("concentrated_topics", []))))
    L.append("  " + t_brief(language, "amp_proxy03", n_bursts=p3.get("burst_count", "—"), sigma=p3.get("sigma", 3)))
    L.append("  " + t_brief(language, "amp_proxy04", dup_pct=p4.get("duplicate_pct", "—")))
    conf_str = coord.get("coordination_confidence", "—")
    conf_localized = t_brief(language, f"confidence_{conf_str}") if conf_str in ("HIGH", "MODERATE", "LOW") else conf_str
    L.append("  " + t_brief(
        language, "amp_signals",
        triggered=coord.get("triggered_count", 0),
        total=coord.get("total_count", 0),
        confidence=conf_localized,
    ))

    # Section 5 — limitations
    L.append("")
    L.append(BAR)
    L.append(t_brief(language, "section_5"))
    L.append(BAR)
    L.append("")
    for line in _gather_limitations(state, schema, set(caps), language):
        L.append("  " + line)

    # Section 6 — recommendations
    L.append("")
    L.append(BAR)
    L.append(t_brief(language, "section_6"))
    L.append(BAR)
    L.append("")
    for f in sorted(findings, key=lambda x: {"HIGH": 0, "MODERATE": 1, "LOW": 2}[x.confidence.value]):
        conf_localized = t_brief(language, f"confidence_{f.confidence.value}")
        L.append(f"  [{conf_localized}] {f.recommendation}")

    L.append("")
    L.append(BAR)
    L.append(t_brief(language, "brief_end"))
    L.append(BAR)
    return "\n".join(L)


def _gather_limitations(state: GraphState, schema, caps: set[str], language: str) -> list[str]:
    out: list[str] = []
    if "temporal" not in caps:
        out.append(t_brief(language, "lim_no_temporal"))
    if "amplification" not in caps:
        out.append(t_brief(language, "lim_no_engagement"))
    if "author_profile" not in caps:
        out.append(t_brief(language, "lim_no_author"))
    out.append(t_brief(language, "lim_model_sentiment"))
    out.append(t_brief(language, "lim_model_topic"))
    out.append(t_brief(language, "lim_causal_proxies"))
    out.append(t_brief(language, "lim_causal_topic_sent"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Node entry
# ─────────────────────────────────────────────────────────────────────────────

def brief_node(state: GraphState) -> dict:
    topic = state.artifacts.get(Stage.TOPIC)
    narrative = state.artifacts.get(Stage.NARRATIVE)
    amp = state.artifacts.get(Stage.AMPLIFICATION)

    if topic is None or amp is None:
        return _skip(state, "missing topic or amplification artifact")
    if topic.summary_stats.get("skipped"):
        return _skip(state, "topic stage was skipped — no clusters to brief on")

    df = pd.read_parquet(topic.parquet_path) if topic.parquet_path else pd.DataFrame()
    matrix = _topic_sentiment_matrix(df)
    narratives_path = (narrative.summary_stats.get("narratives_path") if narrative else None) if narrative else None
    narratives: list[dict] = []
    if narratives_path and Path(narratives_path).exists():
        narratives = json.loads(Path(narratives_path).read_text(encoding="utf-8"))

    language = (state.confirmed_schema.language if state.confirmed_schema else "en") or "en"
    findings = _build_findings(matrix, narratives, amp.summary_stats, language)
    text = _render_brief(state, findings, matrix, narratives, amp.summary_stats, df, language)

    out_path = _out_dir(state) / "analytic_brief.txt"
    out_path.write_text(text, encoding="utf-8")

    summary = {
        "brief_path": str(out_path),
        "language": language,
        "finding_count": len(findings),
        "confidence_breakdown": {
            c.value: sum(1 for f in findings if f.confidence == c) for c in Confidence
        },
    }
    artifact = StageArtifact(
        stage=Stage.BRIEF,
        parquet_path=None,
        summary_stats=summary,
        notes=f"wrote {len(findings)}-finding brief ({language}) to {out_path.name}",
    )
    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.BRIEF,
        action="generated integrated analytic brief",
        affected_rows=len(df),
        reason=f"{len(findings)} findings, language={language}",
        artifact_path=str(out_path),
    )
    return {
        "artifacts": {**state.artifacts, Stage.BRIEF: artifact},
        "audit_log": [*state.audit_log, audit],
        "completed_stages": [*state.completed_stages, Stage.BRIEF],
        "brief_path": str(out_path),
        "findings": findings,
    }


def _skip(state: GraphState, reason: str) -> dict:
    skip_artifact = StageArtifact(
        stage=Stage.BRIEF,
        summary_stats={"skipped": True, "reason": reason},
        notes=f"skipped — {reason}",
    )
    skip_audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.BRIEF,
        action="skipped brief stage",
        reason=reason,
    )
    return {
        "artifacts": {**state.artifacts, Stage.BRIEF: skip_artifact},
        "audit_log": [*state.audit_log, skip_audit],
        "completed_stages": [*state.completed_stages, Stage.BRIEF],
    }
