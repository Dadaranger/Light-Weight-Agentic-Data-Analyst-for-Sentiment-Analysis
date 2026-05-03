"""Brief stage — produce the final BLUF-format integrated analytic brief.

Reads narrative + amplification + topic + sentiment artifacts, picks 2-3
findings, and renders a text file matching the structure in
`brief_writer.md`. Template-based — LLM is not required for the rendering
(only for narrative extraction in the prior stage).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ada.config import settings
from ada.state import (
    AuditEntry,
    BLUFFinding,
    Confidence,
    GraphState,
    Stage,
    StageArtifact,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _out_dir(state: GraphState) -> Path:
    out = settings.project_path(state.project_name) / "artifacts" / "brief" / state.run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _topic_sentiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Per-topic sentiment %s — the table the brief Section 2 cites."""
    if "analyst_label" not in df.columns or "final_label" not in df.columns:
        return pd.DataFrame()
    cross = (
        pd.crosstab(df["analyst_label"], df["final_label"], normalize="index").mul(100).round(1)
    )
    return cross


def _negative_share(matrix: pd.DataFrame) -> pd.Series:
    """Sum of NEGATIVE + NEGATIVE-DISTRESS columns per topic."""
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


def _build_findings(
    matrix: pd.DataFrame,
    narratives: list[dict],
    amp_summary: dict,
    schema_caps: set[str],
) -> list[BLUFFinding]:
    """Pick top findings — highest negative narrative, highest positive narrative,
    and one amplification finding if signals warrant.
    """
    findings: list[BLUFFinding] = []
    coord = amp_summary.get("coordination", {})
    p1 = amp_summary.get("proxy_01_engagement", {})
    p4 = amp_summary.get("proxy_04_duplication", {})

    # Finding 1 — highest-risk negative narrative
    neg = _negative_share(matrix)
    if not neg.empty and neg.iloc[0] > 0:
        topic = neg.index[0]
        pct = float(neg.iloc[0])
        narr = _find_narrative(narratives, topic)
        narr_stmt = (narr or {}).get("NARRATIVE_STATEMENT", "（no narrative extracted）")
        actor = (narr or {}).get("ACTOR", "—")
        post_count = int((narr or {}).get("post_count", 0))
        findings.append(BLUFFinding(
            title=f"Highest-risk narrative: {topic}",
            bluf=(
                f"The narrative cluster '{topic}' carries the highest concentration "
                f"of negative sentiment ({pct:.1f}%) across {post_count:,} posts."
            ),
            evidence=(
                f"Per-topic sentiment matrix: {pct:.1f}% NEGATIVE+DISTRESS in this cluster "
                f"vs. dataset average. Narrative summary: {narr_stmt[:200]}"
            ),
            confidence=Confidence.MODERATE,
            confidence_reason=(
                "Topic + sentiment signals are HIGH; narrative framing requires "
                "human confirmation (currently template-based unless LLM was available)."
            ),
            limitations=[
                "Per-topic sentiment %s reflect a baseline lexicon; nuance and irony may be misclassified.",
                "Cluster definitions come from BERTopic with multilingual MiniLM embeddings; "
                "small clusters or merged topics can shift these numbers.",
            ],
            recommendation=(
                f"Treat '{topic}' as the primary risk narrative for follow-up. "
                f"Verify the actor ({actor}) and review the narrative statement before publishing."
            ),
            related_topics=[topic],
        ))

    # Finding 2 — highest positive (counter-narrative)
    pos = _positive_share(matrix)
    if not pos.empty and pos.iloc[0] > 0:
        topic = pos.index[0]
        pct = float(pos.iloc[0])
        narr = _find_narrative(narratives, topic)
        narr_stmt = (narr or {}).get("NARRATIVE_STATEMENT", "（no narrative extracted）")
        post_count = int((narr or {}).get("post_count", 0))
        findings.append(BLUFFinding(
            title=f"Strongest counter-narrative: {topic}",
            bluf=(
                f"The narrative cluster '{topic}' shows the highest positive-sentiment share "
                f"({pct:.1f}%) across {post_count:,} posts — a candidate for the "
                "counter-narrative section of any communications response."
            ),
            evidence=(
                f"Per-topic sentiment matrix: {pct:.1f}% POSITIVE in this cluster. "
                f"Narrative summary: {narr_stmt[:200]}"
            ),
            confidence=Confidence.MODERATE,
            confidence_reason=(
                "POSITIVE share signal is reliable; specific framing of the counter-narrative "
                "requires human review before public use."
            ),
            limitations=[
                "POSITIVE labels can include sarcasm not caught by the sarcasm regex.",
                "High-positive narrative may include hashtag-driven artifacts (e.g. #台灣加油).",
            ],
            recommendation=(
                f"Use '{topic}' as the foundation for any positive-framing communications. "
                "Hand-pick representative quotes; do not auto-aggregate."
            ),
            related_topics=[topic],
        ))

    # Finding 3 — amplification / coordination, if any signals fired
    triggered = coord.get("triggered_count", 0)
    if triggered >= 1:
        conf_str = coord.get("coordination_confidence", "LOW")
        conf_enum = {"HIGH": Confidence.HIGH, "MODERATE": Confidence.MODERATE, "LOW": Confidence.LOW}[conf_str]
        signals_summary = "; ".join(
            f"{s['name']} {'✓' if s['triggered'] else '✗'}"
            for s in coord.get("signals", [])
            if "not available" not in s["evidence"]
        )
        findings.append(BLUFFinding(
            title=f"Amplification signal: {conf_str.lower()} confidence",
            bluf=(
                f"{triggered}/{coord.get('total_count')} coordination proxies fired; "
                f"engagement concentration is {p1.get('top5_share_pct', '?')}% (top 5%) "
                f"and content duplication is {p4.get('duplicate_pct', '?')}%."
            ),
            evidence=f"Signal checklist: {signals_summary}",
            confidence=conf_enum,
            confidence_reason=coord.get("confidence_reason", ""),
            limitations=[
                "All four proxies are surrogates — no follower graph, no cross-platform "
                "identity resolution, no per-account history available.",
                "Account-level timing intervals and cross-platform synchrony cannot be "
                "evaluated with this dataset; coordination cannot be definitively asserted.",
            ],
            recommendation=(
                "Treat as a preliminary signal worth investigating with platform "
                "trust-and-safety teams. Do not act publicly on coordination claims "
                "without account-level evidence."
            ),
            related_topics=[],
        ))

    return findings


def _render_brief(
    state: GraphState,
    findings: list[BLUFFinding],
    matrix: pd.DataFrame,
    narratives: list[dict],
    amp_summary: dict,
    df: pd.DataFrame,
) -> str:
    """Render the six-section BLUF brief as a plain-text block."""
    today = date.today().isoformat()
    n_rows = len(df)
    project = state.project_name
    schema = state.confirmed_schema
    caps = sorted(schema.capabilities()) if schema else []
    coord = amp_summary.get("coordination", {})
    p1 = amp_summary.get("proxy_01_engagement", {})
    p2 = amp_summary.get("proxy_02_author", {})
    p3 = amp_summary.get("proxy_03_temporal", {})
    p4 = amp_summary.get("proxy_04_duplication", {})

    lines: list[str] = []
    L = lines.append
    BAR = "━" * 70
    L("╔" + "═" * 70 + "╗")
    L(f"║{('  Integrated Analytic Brief — ' + project).ljust(70)}║")
    L("╚" + "═" * 70 + "╝")
    L("")
    L(f"Run ID:    {state.run_id}")
    L(f"Project:   {project}")
    L(f"Date:      {today}")
    L(f"Dataset:   {Path(state.raw_file_path).name}, {n_rows:,} rows")
    L(f"Capabilities used: {caps}")
    L("Status:    Draft — pending peer review")
    L("")

    # Section 1 — findings
    L(BAR)
    L("Section 1 — Key findings")
    L(BAR)
    if not findings:
        L("  (No findings produced — insufficient data or skipped upstream stages.)")
    for i, f in enumerate(findings, 1):
        L("")
        L(f"[{i}] {f.title}")
        L(f"    BLUF       : {f.bluf}")
        L(f"    EVIDENCE   : {f.evidence}")
        L(f"    CONFIDENCE : {f.confidence.value} — {f.confidence_reason}")
        for j, lim in enumerate(f.limitations, 1):
            L(f"    LIMITATION {j}: {lim}")
        L(f"    RECOMMEND  : {f.recommendation}")

    # Section 2 — sentiment by topic (never aggregated)
    L("")
    L(BAR)
    L("Section 2 — Sentiment distribution (per topic, never aggregated)")
    L(BAR)
    L("")
    if matrix.empty:
        L("  (Topic × sentiment matrix unavailable.)")
    else:
        for topic in matrix.head(20).index:
            row = matrix.loc[topic]
            pos = row.get("POSITIVE", 0.0)
            neg = row.get("NEGATIVE", 0.0)
            dst = row.get("NEGATIVE-DISTRESS", 0.0)
            ntr = row.get("NEUTRAL", 0.0)
            n_topic = int((df["analyst_label"] == topic).sum()) if "analyst_label" in df.columns else 0
            L(
                f"  {str(topic)[:32]:<32} | pos {pos:5.1f}% | "
                f"neg {neg:5.1f}% | dst {dst:5.1f}% | ntr {ntr:5.1f}% | n={n_topic:,}"
            )
        L("")
        L("  ⚠ Do not cite an aggregate 'X% negative' figure. Negative sentiment is")
        L("    concentrated in specific narratives — report by narrative.")

    # Section 3 — narratives
    L("")
    L(BAR)
    L("Section 3 — Topic / narrative registry")
    L(BAR)
    L("")
    for n in narratives[:20]:
        L(f"  Topic: {n.get('analyst_label', '—')}  ({n.get('post_count', 0):,} posts)")
        stmt = n.get("NARRATIVE_STATEMENT", "—")
        L(f"  Narrative: {stmt[:240]}")
        L(f"  Actor: {n.get('ACTOR', '—')}  |  Desired reaction: {n.get('DESIRED', '—')}")
        L("")

    # Section 4 — amplification proxies
    L(BAR)
    L("Section 4 — Amplification indicators")
    L(BAR)
    L("")
    L(f"  Proxy 01 engagement concentration : top 5% carry "
      f"{p1.get('top5_share_pct', '—')}% of total engagement")
    L(f"  Proxy 02 author-type concentration: "
      f"{len(p2.get('concentrated_topics', []))} topics > 70% single author type")
    L(f"  Proxy 03 temporal bursts          : {p3.get('burst_count', '—')} bursts above "
      f"{p3.get('sigma', 3)}σ")
    L(f"  Proxy 04 content duplication      : {p4.get('duplicate_pct', '—')}% duplicate rate")
    L(f"  Coordination signal checklist     : {coord.get('triggered_count', 0)}/"
      f"{coord.get('total_count', 0)} → confidence {coord.get('coordination_confidence', '—')}")

    # Section 5 — limitations
    L("")
    L(BAR)
    L("Section 5 — Analytic limitations")
    L(BAR)
    L("")
    lims = _gather_limitations(state, schema, caps)
    for cat, lim in lims:
        L(f"  [{cat}] {lim}")

    # Section 6 — recommendations
    L("")
    L(BAR)
    L("Section 6 — Recommendations")
    L(BAR)
    L("")
    for f in sorted(findings, key=lambda x: {"HIGH": 0, "MODERATE": 1, "LOW": 2}[x.confidence.value]):
        L(f"  [{f.confidence.value}] {f.recommendation}")

    L("")
    L(BAR)
    L("Brief end — draft, awaiting peer review")
    L(BAR)
    return "\n".join(lines)


def _gather_limitations(state: GraphState, schema, caps) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if "temporal" not in caps:
        out.append(("data coverage", "no timestamp column — temporal analyses (Proxy 03, phase analysis) skipped"))
    if "amplification" not in caps:  # capabilities() returns 'amplification' for engagement_col
        out.append(("data coverage", "no engagement column — Proxy 01 (engagement concentration) skipped"))
    if "author_profile" not in caps:
        out.append(("data coverage", "no author column — Proxy 02 (author concentration) skipped"))
    out.append((
        "model",
        "Sentiment uses a Tier 1 + Tier 2 baseline (rules + lexicon, ~120 zh-TW seed terms); "
        "no transformer model. Subtle context-dependent sentiment can be misclassified.",
    ))
    out.append((
        "model",
        "Topic clustering uses BERTopic with multilingual MiniLM embeddings; outlier rate "
        "and topic count are sensitive to min_topic_size and nr_topics — re-tune via "
        "domain_knowledge.thresholds before publishing final percentages.",
    ))
    out.append((
        "causal",
        "All four amplification indicators are PROXIES, not measurements. The dataset "
        "lacks follower graphs, cross-platform identity, and account history — "
        "coordination cannot be definitively asserted from these signals alone.",
    ))
    out.append((
        "causal",
        "Per-topic sentiment %s do not establish causation. High-engagement clusters may "
        "reflect algorithmic amplification rather than organic resonance.",
    ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Node entry point
# ─────────────────────────────────────────────────────────────────────────────

def brief_node(state: GraphState) -> dict:
    topic = state.artifacts.get(Stage.TOPIC)
    narrative = state.artifacts.get(Stage.NARRATIVE)
    amp = state.artifacts.get(Stage.AMPLIFICATION)

    if topic is None or amp is None:
        # Brief is the terminal stage — degrade gracefully if upstream skipped
        skip_artifact = StageArtifact(
            stage=Stage.BRIEF,
            summary_stats={"skipped": True, "reason": "missing topic or amplification artifact"},
            notes="skipped — upstream artifacts missing",
        )
        skip_audit = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            stage=Stage.BRIEF,
            action="skipped brief stage",
            reason="missing topic or amplification artifact",
        )
        return {
            "artifacts": {**state.artifacts, Stage.BRIEF: skip_artifact},
            "audit_log": [*state.audit_log, skip_audit],
            "completed_stages": [*state.completed_stages, Stage.BRIEF],
        }

    if topic.summary_stats.get("skipped"):
        return _skip(state, "topic stage was skipped — no clusters to brief on")

    df = pd.read_parquet(topic.parquet_path) if topic.parquet_path else pd.DataFrame()
    matrix = _topic_sentiment_matrix(df)
    narratives_path = (narrative.summary_stats.get("narratives_path") if narrative else None) if narrative else None
    narratives: list[dict] = []
    if narratives_path and Path(narratives_path).exists():
        narratives = json.loads(Path(narratives_path).read_text(encoding="utf-8"))

    schema_caps = (state.confirmed_schema.capabilities() if state.confirmed_schema else set())
    findings = _build_findings(matrix, narratives, amp.summary_stats, schema_caps)
    text = _render_brief(state, findings, matrix, narratives, amp.summary_stats, df)

    out_path = _out_dir(state) / "analytic_brief.txt"
    out_path.write_text(text, encoding="utf-8")

    summary = {
        "brief_path": str(out_path),
        "finding_count": len(findings),
        "confidence_breakdown": {
            c.value: sum(1 for f in findings if f.confidence == c)
            for c in Confidence
        },
    }

    artifact = StageArtifact(
        stage=Stage.BRIEF,
        parquet_path=None,
        summary_stats=summary,
        notes=f"wrote {len(findings)}-finding brief to {out_path.name}",
    )

    audit = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage=Stage.BRIEF,
        action="generated integrated analytic brief",
        affected_rows=len(df),
        reason=f"{len(findings)} findings",
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
