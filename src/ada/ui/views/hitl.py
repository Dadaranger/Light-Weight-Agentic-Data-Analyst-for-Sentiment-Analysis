"""HITL page — renders the appropriate form based on the pending question.

Three question types currently supported:
  - schema_infer / confirm   → editable column-role mapping
  - topic / label            → per-cluster label form (the headline UX)
  - sentiment / calibrate    → 10-row labeling table
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ada.i18n import t_ui
from ada.ui import state as ui_state


def render() -> None:
    lang = st.session_state.language
    st.title(t_ui(lang, "hitl_header"))

    payload = ui_state.get_pending_interrupt()
    if payload is None:
        st.info(t_ui(lang, "hitl_no_question"))
        if ui_state.is_done():
            st.success(t_ui(lang, "progress_done"))
            if st.button("→ " + t_ui(lang, "nav_report"), type="primary"):
                ui_state.goto(ui_state.PAGE_REPORT)
                st.rerun()
        return

    # Why-asking note
    why = payload.get("why_asking", "")
    if why:
        st.caption(f"💡 {t_ui(lang, 'hitl_why')}: {why}")

    qtype = payload.get("type")
    stage = payload.get("stage")
    if stage == "schema_infer" and qtype == "confirm":
        _schema_form(payload, lang)
    elif stage == "topic" and qtype == "label":
        _topic_label_form(payload, lang)
    elif stage == "sentiment" and qtype == "calibrate":
        _sentiment_calibrate_form(payload, lang)
    else:
        # Generic fallback — show JSON, accept proposal
        _generic_form(payload, lang)


# ─────────────────────────────────────────────────────────────────────────────
# Schema confirm
# ─────────────────────────────────────────────────────────────────────────────

def _schema_form(payload: dict, lang: str) -> None:
    st.subheader(t_ui(lang, "hitl_schema_title"))
    st.write(t_ui(lang, "hitl_schema_intro"))

    proposal = payload.get("proposal", {})
    proposed_schema = proposal.get("schema", {})
    col_profiles = payload.get("payload", {}).get("column_profiles", [])
    col_names = ["（無 / none）"] + [c["name"] for c in col_profiles]
    none_marker = col_names[0]

    def _index_for(role_value):
        if role_value is None or role_value == "":
            return 0
        try:
            return col_names.index(role_value)
        except ValueError:
            return 0

    st.divider()
    st.caption(t_ui(lang, "hitl_schema_samples_caption"))
    if col_profiles:
        sample_df = pd.DataFrame({
            c["name"]: c.get("sample_values", []) + [""] * (5 - len(c.get("sample_values", [])))
            for c in col_profiles
        }).head(5)
        st.dataframe(sample_df, hide_index=True, use_container_width=True)

    st.divider()
    cols = st.columns(2)
    fields: dict[str, str | None] = {}
    role_order = [
        ("text_col", "hitl_schema_role_text", True),
        ("id_col", "hitl_schema_role_id", True),
        ("timestamp_col", "hitl_schema_role_ts", False),
        ("author_col", "hitl_schema_role_author", False),
        ("engagement_col", "hitl_schema_role_engagement", False),
        ("platform_col", "hitl_schema_role_platform", False),
    ]
    for i, (key, label_key, _required) in enumerate(role_order):
        with cols[i % 2]:
            chosen = st.selectbox(
                t_ui(lang, label_key),
                options=col_names,
                index=_index_for(proposed_schema.get(key)),
            )
            fields[key] = None if chosen == none_marker else chosen

    language = st.text_input(
        t_ui(lang, "hitl_schema_language"),
        value=proposed_schema.get("language", "zh-TW"),
        max_chars=10,
    )

    st.divider()
    submit_col, _ = st.columns([1, 3])
    with submit_col:
        if st.button(t_ui(lang, "hitl_submit"), type="primary", use_container_width=True):
            new_schema = {
                "id_col": fields["id_col"],
                "text_col": fields["text_col"],
                "language": language,
                "timestamp_col": fields["timestamp_col"],
                "author_col": fields["author_col"],
                "engagement_col": fields["engagement_col"],
                "platform_col": fields["platform_col"],
                "extra_dims": proposed_schema.get("extra_dims", {}),
            }
            response = {"schema": new_schema, "approved": True}
            with st.spinner("..."):
                ui_state.resume_with(response)
            _redirect_after_resume()


# ─────────────────────────────────────────────────────────────────────────────
# Topic label — the headline labeling UX
# ─────────────────────────────────────────────────────────────────────────────

def _topic_label_form(payload: dict, lang: str) -> None:
    body = payload.get("payload", {})
    proposal = payload.get("proposal", {})
    clusters = body.get("clusters", [])
    proposed_labels: dict[str, str] = dict(proposal.get("labels", {}))

    st.subheader(t_ui(lang, "hitl_topic_title"))
    st.write(t_ui(lang, "hitl_topic_intro", n=len(clusters)))
    st.caption(
        f"離群值：{body.get('outlier_count', 0)} 筆 ({body.get('outlier_pct', 0):.1f}%)"
    )

    st.divider()

    # Form widgets — collect into session-state-backed dict so the user can
    # scroll without losing entries
    if "topic_label_inputs" not in st.session_state:
        st.session_state.topic_label_inputs = dict(proposed_labels)

    save_to_memory = st.checkbox(t_ui(lang, "hitl_topic_save_to_memory"))

    with st.form("topic_labels_form", clear_on_submit=False):
        for cluster in clusters:
            tid = cluster["topic_id"]
            tid_str = str(tid)
            st.markdown(
                f"### 🎯 {t_ui(lang, 'hitl_topic_cluster')} {tid} "
                f"· {cluster.get('size', 0)} {t_ui(lang, 'hitl_topic_size')}"
            )
            st.markdown(
                f"**{t_ui(lang, 'hitl_topic_keywords')}:** "
                f"`{' · '.join(cluster.get('keywords', [])[:10])}`"
            )

            # Editable label input
            current = st.session_state.topic_label_inputs.get(tid_str, proposed_labels.get(tid_str, ""))
            new_label = st.text_input(
                t_ui(lang, "hitl_topic_label_input"),
                value=current,
                key=f"label_input_{tid_str}",
            )
            st.session_state.topic_label_inputs[tid_str] = new_label

            # Sample posts
            samples = cluster.get("samples", [])
            if samples:
                with st.expander(t_ui(lang, "hitl_topic_samples"), expanded=False):
                    for s in samples:
                        sentiment = s.get("sentiment", "")
                        sent_color = {
                            "POSITIVE": "🟢", "NEGATIVE": "🔴",
                            "NEGATIVE-DISTRESS": "🟣", "NEUTRAL": "⚪",
                            "UNCERTAIN": "🟡",
                        }.get(sentiment, "⚫")
                        st.markdown(
                            f"{sent_color} **{s.get('platform', '')}** "
                            f"(互動 {int(s.get('engagement', 0))}) — {s.get('text', '')}"
                        )
            st.divider()

        submitted = st.form_submit_button(t_ui(lang, "hitl_submit"), type="primary", use_container_width=True)

    if submitted:
        labels = dict(st.session_state.topic_label_inputs)
        # Always include outlier label
        labels.setdefault("-1", proposed_labels.get("-1", "未歸類貼文（離群值）"))
        response = {
            "labels": labels,
            "approved": True,
            "persist_to_memory": save_to_memory,
        }
        with st.spinner("..."):
            ui_state.resume_with(response)
        # Clear input cache so next run gets fresh proposals
        st.session_state.pop("topic_label_inputs", None)
        _redirect_after_resume()


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment calibrate
# ─────────────────────────────────────────────────────────────────────────────

def _sentiment_calibrate_form(payload: dict, lang: str) -> None:
    body = payload.get("payload", {})
    items = body.get("items", [])
    choices = body.get("label_choices", t_ui(lang, "hitl_sentiment_label_choices"))

    st.subheader(t_ui(lang, "hitl_sentiment_title"))
    st.write(t_ui(lang, "hitl_sentiment_intro", n=len(items)))
    st.caption(f"目前不一致率：{body.get('disagreement_pct', 0)}%")

    st.divider()

    if "sentiment_calibrate_inputs" not in st.session_state:
        st.session_state.sentiment_calibrate_inputs = {
            it["id"]: it.get("final_label", "NEUTRAL") for it in items
        }

    with st.form("sentiment_calibrate_form", clear_on_submit=False):
        for item in items:
            iid = item["id"]
            with st.container(border=True):
                st.markdown(f"**{item.get('text', '')}**")
                st.caption(
                    f"agent labels: T1={item.get('t1_label', '?')} ({item.get('t1_conf', 0):.2f}) "
                    f"· T2={item.get('t2_label', '?')} ({item.get('t2_conf', 0):.2f}) "
                    f"· final={item.get('final_label', '?')}"
                )
                current = st.session_state.sentiment_calibrate_inputs.get(iid, "NEUTRAL")
                idx = choices.index(current) if current in choices else 0
                chosen = st.radio(
                    label=f"label_{iid}",
                    options=choices,
                    index=idx,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"sent_choice_{iid}",
                )
                st.session_state.sentiment_calibrate_inputs[iid] = chosen

        notes = st.text_area(t_ui(lang, "hitl_sentiment_notes"), height=80)
        submitted = st.form_submit_button(t_ui(lang, "hitl_submit"), type="primary", use_container_width=True)

    if submitted:
        labels = [
            {"id": iid, "label": lbl}
            for iid, lbl in st.session_state.sentiment_calibrate_inputs.items()
        ]
        response = {"labels": labels, "approved": True, "notes": notes}
        with st.spinner("..."):
            ui_state.resume_with(response)
        st.session_state.pop("sentiment_calibrate_inputs", None)
        _redirect_after_resume()


# ─────────────────────────────────────────────────────────────────────────────
# Generic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generic_form(payload: dict, lang: str) -> None:
    st.warning(f"未知的問題類型 / Unknown question type: stage={payload.get('stage')} type={payload.get('type')}")
    st.write(payload.get("prompt", ""))
    st.json(payload.get("payload", {}), expanded=False)
    st.markdown("**Proposal:**")
    st.json(payload.get("proposal", {}), expanded=True)
    if st.button(t_ui(lang, "hitl_submit"), type="primary"):
        with st.spinner("..."):
            ui_state.resume_with(payload.get("proposal") or {"approved": True})
        _redirect_after_resume()


# ─────────────────────────────────────────────────────────────────────────────
# Post-resume routing
# ─────────────────────────────────────────────────────────────────────────────

def _redirect_after_resume() -> None:
    if ui_state.get_pending_interrupt():
        ui_state.goto(ui_state.PAGE_HITL)
    elif ui_state.is_done():
        ui_state.goto(ui_state.PAGE_REPORT)
    else:
        ui_state.goto(ui_state.PAGE_PROGRESS)
    st.rerun()
