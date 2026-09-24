"""
ui/upload_page.py — upload flow: CSV upload -> load -> clean -> classify ->
show inferred schema with manual override controls -> stash results in
session state for the dashboard views to consume.
"""

from __future__ import annotations

import streamlit as st

from ingestion.classifier import (
    ROLE_DATE,
    ROLE_DIMENSION,
    ROLE_ID,
    ROLE_METRIC,
    ROLE_TEXT,
    ambiguous_columns,
    apply_overrides,
    build_llm_classification_payload,
    classify_columns,
)
from ingestion.cleaner import clean_dataframe
from ingestion.loader import load_csv, load_file, sanitize_filename
from insight.client import BudgetExhaustedError, classify_ambiguous_columns

ALL_ROLES = [ROLE_DATE, ROLE_METRIC, ROLE_DIMENSION, ROLE_ID, ROLE_TEXT]


def render_upload_page() -> bool:
    """Renders the upload UI. Returns True once a dataset is loaded and
    classified and ready for the dashboard tabs to use."""
    st.header("Upload your data")
    st.caption(
        "Your raw data never leaves this server — only small, aggregated "
        "summaries are ever sent to the AI insight layer."
    )

    import hashlib
    from insight.cache import clear_view_cache

    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return "dataset" in st.session_state

    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    safe_name = sanitize_filename(uploaded.name)

    if st.session_state.get("_uploaded_file_hash") != file_hash:
        load_result = load_file(file_bytes, filename=safe_name)

        if not load_result.success:
            st.error(f"Could not read this file: {load_result.error}")
            return False

        for note in load_result.notes:
            st.caption(f"ℹ️ {note}")

        cleaned_df, clean_report = clean_dataframe(load_result.dataframe)
        for msg in clean_report.messages:
            st.caption(f"🧹 {msg}")

        classifications = classify_columns(cleaned_df)

        st.session_state["dataset"] = cleaned_df
        st.session_state["classifications"] = {c.name: c for c in classifications}
        st.session_state["_uploaded_filename"] = safe_name
        st.session_state["_uploaded_file_hash"] = file_hash
        st.session_state["dataset_name"] = safe_name
        st.session_state["schema_confirmed"] = False
        st.session_state.pop("overrides", None)
        st.session_state.pop("_export_figures", None)
        st.session_state.pop("_pptx_buffer", None)
        st.session_state.pop("_view_insights", None)
        clear_view_cache()

    df = st.session_state["dataset"]
    classifications = list(st.session_state["classifications"].values())

    st.subheader("Detected schema")
    st.caption("Review and override any column's role below, then continue to the dashboard.")

    overrides = st.session_state.get("overrides", {})
    ambiguous = {c.name for c in ambiguous_columns(classifications)}

    for c in classifications:
        cols = st.columns([3, 2, 2, 3])
        cols[0].write(c.name)
        flag = " ⚠️ low confidence" if c.name in ambiguous else ""
        cols[1].write(f"{c.role}{flag}")
        cols[2].write(f"{c.confidence:.0%} confidence")
        current_role = overrides.get(c.name, c.role)
        new_role = cols[3].selectbox(
            "Override", ALL_ROLES, index=ALL_ROLES.index(current_role), key=f"override_{c.name}", label_visibility="collapsed"
        )
        if new_role != c.role:
            overrides[c.name] = new_role

    st.session_state["overrides"] = overrides

    if ambiguous:
        if st.button("✨ Auto-classify ambiguous columns with AI", key="auto_classify_ai_btn"):
            ambiguous_objs = [c for c in classifications if c.name in ambiguous]
            payload = build_llm_classification_payload(df, ambiguous_objs)
            try:
                llm_results = classify_ambiguous_columns(payload)
                for item in llm_results:
                    col_name = item.get("column_name")
                    role = item.get("role")
                    if col_name in ambiguous and role in ALL_ROLES:
                        overrides[col_name] = role
                st.session_state["overrides"] = overrides
                st.rerun()
            except BudgetExhaustedError:
                st.info("AI classification budget reached — keeping rule-based guesses.")
            except Exception as exc:
                st.warning(f"Could not run AI classification: {exc}")

    if st.button("Confirm schema and build dashboard", type="primary", key="confirm_schema_btn"):
        final_classifications = apply_overrides(classifications, overrides)
        st.session_state["classifications"] = {c.name: c for c in final_classifications}
        st.session_state["schema_confirmed"] = True

    with st.expander("Preview raw data"):
        st.dataframe(df.head(50))

    return bool(st.session_state.get("schema_confirmed"))
