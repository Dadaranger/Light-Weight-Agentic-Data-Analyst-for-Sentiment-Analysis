"""Start page — upload a file, set project name, kick off the pipeline."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ada.config import settings
from ada.i18n import t_ui
from ada.ui import state as ui_state


def render() -> None:
    lang = st.session_state.language
    st.title(t_ui(lang, "app_title"))
    st.caption(t_ui(lang, "app_subtitle"))
    st.divider()

    st.header(t_ui(lang, "start_header"))
    st.write(t_ui(lang, "start_intro"))

    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded = st.file_uploader(
            label=t_ui(lang, "start_file_label"),
            help=t_ui(lang, "start_file_help"),
            type=["csv", "xlsx", "xls", "json", "jsonl", "parquet"],
        )
    with col_b:
        project = st.text_input(
            label=t_ui(lang, "start_project_label"),
            value="",
            help=t_ui(lang, "start_project_help"),
        )

    user_prompt = st.text_area(
        label=t_ui(lang, "start_prompt_label"),
        help=t_ui(lang, "start_prompt_help"),
        max_chars=400,
        height=80,
    )

    st.divider()
    if st.button(t_ui(lang, "start_run_button"), type="primary", use_container_width=True):
        if uploaded is None:
            st.error(t_ui(lang, "start_no_file"))
            return
        if not project.strip():
            st.error("請輸入專案名稱 / Please enter a project name")
            return

        # Persist the upload to the project's data dir
        proj_dir = settings.project_path(project.strip()) / "data"
        proj_dir.mkdir(parents=True, exist_ok=True)
        target = proj_dir / uploaded.name
        target.write_bytes(uploaded.getvalue())

        with st.spinner("管線執行中..."):
            ui_state.start_run(target, project.strip(), user_prompt.strip())

        # Decide where to land next
        if ui_state.get_pending_interrupt():
            ui_state.goto(ui_state.PAGE_HITL)
        elif ui_state.is_done():
            ui_state.goto(ui_state.PAGE_REPORT)
        else:
            ui_state.goto(ui_state.PAGE_PROGRESS)
        st.rerun()
