"""Report page — render the BLUF brief, EDA figures, topic-sentiment heatmap."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ada.i18n import t_ui
from ada.state import Stage
from ada.ui import state as ui_state


def render() -> None:
    lang = st.session_state.language
    st.title(t_ui(lang, "report_header"))

    values = ui_state.get_state_values()
    if not values:
        st.info(t_ui(lang, "progress_no_run"))
        return

    artifacts = values.get("artifacts", {})
    brief_art = _get_artifact(artifacts, Stage.BRIEF)
    if brief_art is None or _summary(brief_art).get("skipped"):
        st.info(t_ui(lang, "report_no_brief"))
        return

    summary = _summary(brief_art)
    brief_path = Path(summary.get("brief_path", ""))
    if not brief_path.exists():
        st.error(f"簡報檔案不存在 / Brief file missing: {brief_path}")
        return

    # Run metadata
    with st.container(border=True):
        st.subheader(t_ui(lang, "report_run_meta"))
        meta_cols = st.columns(4)
        meta_cols[0].metric("執行 ID", values.get("run_id", "—"))
        meta_cols[1].metric("專案", values.get("project_name", "—"))
        meta_cols[2].metric("發現數", summary.get("finding_count", 0))
        meta_cols[3].metric("簡報語言", summary.get("language", "—"))

    # Download
    text = brief_path.read_text(encoding="utf-8")
    st.download_button(
        t_ui(lang, "report_download"),
        data=text.encode("utf-8"),
        file_name=brief_path.name,
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()

    # Topic × sentiment matrix as a heatmap
    topic_art = _get_artifact(artifacts, Stage.TOPIC)
    if topic_art is not None:
        topic_parquet = _attr(topic_art, "parquet_path")
        if topic_parquet and Path(topic_parquet).exists():
            st.subheader(t_ui(lang, "report_matrix"))
            df = pd.read_parquet(topic_parquet)
            if "analyst_label" in df.columns and "final_label" in df.columns:
                cross = (
                    pd.crosstab(df["analyst_label"], df["final_label"], normalize="index")
                    .mul(100).round(1)
                )
                # Display as styled dataframe — red for negative, green for positive
                styled = cross.style.background_gradient(
                    cmap="RdYlGn_r",
                    subset=[c for c in ("NEGATIVE", "NEGATIVE-DISTRESS") if c in cross.columns],
                ).background_gradient(
                    cmap="RdYlGn",
                    subset=[c for c in ("POSITIVE",) if c in cross.columns],
                ).format("{:.1f}%")
                st.dataframe(styled, use_container_width=True, height=600)

    st.divider()

    # EDA figures
    eda_art = _get_artifact(artifacts, Stage.EDA)
    if eda_art is not None:
        figs = _attr(eda_art, "figure_paths") or []
        if figs:
            st.subheader(t_ui(lang, "report_figures"))
            for fpath in figs:
                if Path(fpath).exists():
                    st.image(fpath, use_container_width=True)

    st.divider()

    # Full brief text
    st.subheader(t_ui(lang, "report_full_text"))
    st.code(text, language=None, wrap_lines=False)


def _get_artifact(artifacts: dict, stage: Stage):
    return artifacts.get(stage) or artifacts.get(stage.value)


def _attr(obj, name):
    return getattr(obj, name, None) if hasattr(obj, name) else (
        obj.get(name) if isinstance(obj, dict) else None
    )


def _summary(art) -> dict:
    return _attr(art, "summary_stats") or {}
