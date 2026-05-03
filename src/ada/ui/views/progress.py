"""Progress page — show stage status, audit log, recent artifacts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ada.i18n import stage_label, t_ui
from ada.state import Stage
from ada.ui import state as ui_state


_STAGE_ORDER = [
    Stage.INGEST, Stage.SCHEMA_INFER, Stage.RESHAPE, Stage.EDA, Stage.CLEAN,
    Stage.PREPROCESS, Stage.SENTIMENT, Stage.TOPIC, Stage.NARRATIVE,
    Stage.AMPLIFICATION, Stage.BRIEF,
]


def _normalize_stage(s):
    return s.value if hasattr(s, "value") else str(s)


def render() -> None:
    lang = st.session_state.language
    st.title(t_ui(lang, "progress_header"))

    if not st.session_state.run_id:
        st.info(t_ui(lang, "progress_no_run"))
        return

    values = ui_state.get_state_values()
    if not values:
        st.warning("尚無狀態資料 / No state available")
        return

    completed = {_normalize_stage(s) for s in values.get("completed_stages", [])}

    # Status banner
    if ui_state.get_pending_interrupt():
        st.warning(t_ui(lang, "progress_paused"))
        if st.button("→ " + t_ui(lang, "nav_hitl"), type="primary"):
            ui_state.goto(ui_state.PAGE_HITL)
            st.rerun()
    elif ui_state.is_done():
        st.success(t_ui(lang, "progress_done"))
        if st.button("→ " + t_ui(lang, "nav_report"), type="primary"):
            ui_state.goto(ui_state.PAGE_REPORT)
            st.rerun()
    else:
        st.info(t_ui(lang, "progress_running"))

    st.divider()

    # Stage progress
    st.subheader(t_ui(lang, "progress_stages"))
    cols = st.columns(len(_STAGE_ORDER))
    for col, stage in zip(cols, _STAGE_ORDER):
        is_done = stage.value in completed
        with col:
            icon = "✅" if is_done else "⏳"
            st.metric(
                label=stage_label(lang, stage.value),
                value=icon,
                delta=None,
                label_visibility="visible",
            )

    st.divider()

    # Audit log
    st.subheader(t_ui(lang, "progress_audit"))
    audit = values.get("audit_log", [])
    if audit:
        rows = []
        for e in audit[-30:]:
            ts = getattr(e, "timestamp", None) or e.get("timestamp")
            stage = getattr(e, "stage", None) or e.get("stage")
            stage_v = _normalize_stage(stage)
            action = getattr(e, "action", None) or e.get("action")
            rows_count = getattr(e, "affected_rows", None) or e.get("affected_rows")
            reason = getattr(e, "reason", None) or e.get("reason")
            rows.append({
                t_ui(lang, "audit_col_time"): str(ts)[11:19] if ts else "",
                t_ui(lang, "audit_col_stage"): stage_label(lang, stage_v),
                t_ui(lang, "audit_col_action"): action or "",
                t_ui(lang, "audit_col_rows"): rows_count if rows_count else "",
                t_ui(lang, "audit_col_reason"): reason or "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("（無稽核紀錄 / no audit entries yet）")

    # Artifacts produced
    st.divider()
    st.subheader(t_ui(lang, "progress_artifacts"))
    artifacts = values.get("artifacts", {})
    if artifacts:
        for stage_key, art in artifacts.items():
            stage_v = _normalize_stage(stage_key)
            with st.expander(f"📦 {stage_label(lang, stage_v)}", expanded=False):
                summary = getattr(art, "summary_stats", None) or art.get("summary_stats", {})
                notes = getattr(art, "notes", None) or art.get("notes", "")
                parquet = getattr(art, "parquet_path", None) or art.get("parquet_path")
                figs = getattr(art, "figure_paths", None) or art.get("figure_paths", [])
                if notes:
                    st.caption(notes)
                if parquet:
                    st.code(parquet, language=None)
                if figs:
                    st.caption(f"📊 {len(figs)} figure(s)")
                    for fpath in figs:
                        if Path(fpath).exists():
                            st.image(fpath, use_container_width=True)
                if summary:
                    st.json(_compact_summary(summary), expanded=False)


def _compact_summary(summary: dict) -> dict:
    """Trim large fields from summary_stats so the JSON view stays readable."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, dict) and len(str(v)) > 800:
            out[k] = f"<{len(v)} keys, truncated>"
        elif isinstance(v, list) and len(v) > 10:
            out[k] = v[:10] + [f"... +{len(v) - 10} more"]
        else:
            out[k] = v
    return out
