"""Streamlit UI entry point.

Run with:
    streamlit run src/ada/ui/app.py

Single-page app with sidebar navigation. Four conceptual pages:
  ① 啟動分析   — upload + run
  ② 進度追蹤   — live audit log + stage progress
  ③ 人工確認   — HITL response forms (schema confirm, topic labels, sentiment calibrate)
  ④ 分析報告   — final BLUF brief
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable when launched directly via `streamlit run`
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from ada.i18n import t_ui
from ada.ui import state as ui_state
from ada.ui.views import hitl, progress, report, start


def _sidebar() -> None:
    lang = st.session_state.language
    st.sidebar.title("🇹🇼 ADA")
    st.sidebar.caption(t_ui(lang, "app_subtitle"))
    st.sidebar.divider()

    # Page selector
    page = st.sidebar.radio(
        label="navigation",
        options=[ui_state.PAGE_START, ui_state.PAGE_PROGRESS, ui_state.PAGE_HITL, ui_state.PAGE_REPORT],
        format_func=lambda p: {
            ui_state.PAGE_START: t_ui(lang, "nav_start"),
            ui_state.PAGE_PROGRESS: t_ui(lang, "nav_progress"),
            ui_state.PAGE_HITL: t_ui(lang, "nav_hitl"),
            ui_state.PAGE_REPORT: t_ui(lang, "nav_report"),
        }[p],
        index=[
            ui_state.PAGE_START, ui_state.PAGE_PROGRESS,
            ui_state.PAGE_HITL, ui_state.PAGE_REPORT,
        ].index(st.session_state.page),
        label_visibility="collapsed",
    )
    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(t_ui(lang, "current_run_label"))
    if st.session_state.run_id:
        st.sidebar.code(f"{st.session_state.run_id}\n{st.session_state.project_name}", language=None)

        # Status indicator
        if ui_state.get_pending_interrupt():
            st.sidebar.warning("⏸ " + t_ui(lang, "progress_paused"))
        elif ui_state.is_done():
            st.sidebar.success("✅ " + t_ui(lang, "progress_done"))
        else:
            st.sidebar.info("▶ " + t_ui(lang, "progress_running"))

        if st.sidebar.button("🔄 重新啟動 / Reset"):
            ui_state.reset()
            st.rerun()
    else:
        st.sidebar.caption(t_ui(lang, "no_active_run"))


def main() -> None:
    st.set_page_config(
        page_title="Agentic Data Analyst",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui_state.init_session()
    _sidebar()

    page = st.session_state.page
    if page == ui_state.PAGE_START:
        start.render()
    elif page == ui_state.PAGE_PROGRESS:
        progress.render()
    elif page == ui_state.PAGE_HITL:
        hitl.render()
    elif page == ui_state.PAGE_REPORT:
        report.render()


if __name__ == "__main__":
    main()
