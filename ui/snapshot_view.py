"""
ui/snapshot_view.py — Cross-Sectional Snapshot Dashboard.

Rendered when a dataset does not have a continuous time axis (e.g. Inventory,
Product Catalogs, Sales Leaderboards, Store Snapshots).
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from analytics.profiler import DatasetProfile
from analytics.snapshot import SnapshotAnalyticsResult
from ui.components import humanize_col, render_kpi_tile, render_insight_card


def render_snapshot_executive_view(
    df: pd.DataFrame,
    snapshot_res: SnapshotAnalyticsResult,
    profile: DatasetProfile,
    model: str,
    allow_paid: bool = False,
) -> None:
    st.header("Executive Summary")
    st.caption(f"Cross-sectional analysis of {snapshot_res.total_items} {profile.entity_dimension or 'items'}.")

    # Top KPI tiles
    kpis = list(snapshot_res.kpis.values())
    if kpis:
        cols = st.columns(min(4, len(kpis)))
        for i, kpi in enumerate(kpis[:4]):
            with cols[i % len(cols)]:
                render_kpi_tile(kpi)

    # Pareto concentration callout
    if snapshot_res.pareto_text:
        st.info(f"💡 **Concentration Insight**: {snapshot_res.pareto_text}")

    # Top 10 Ranking Horizontal Bar Chart
    if snapshot_res.top_entities and profile.primary_metric:
        h_metric = humanize_col(profile.primary_metric)
        h_entity = humanize_col(profile.entity_dimension or "Entity")

        # Reverse so highest is at the top of horizontal bar
        ents = [e.entity for e in reversed(snapshot_res.top_entities)]
        vals = [e.value for e in reversed(snapshot_res.top_entities)]
        pcts = [f"{e.pct_share:.1f}%" for e in reversed(snapshot_res.top_entities)]

        fig = go.Figure(
            go.Bar(
                x=vals,
                y=ents,
                orientation="h",
                marker=dict(color="#1a73e8"),
                text=pcts,
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Top 10 {h_entity}s by {h_metric}",
            xaxis_title=h_metric,
            yaxis_title=h_entity,
            template="plotly_white",
            height=420,
            margin=dict(l=80, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, width="stretch", key="snapshot_exec_top_chart")
        st.session_state.setdefault("_export_figures", {})["Executive_trend"] = fig

    # Insight card
    facts_packet = {
        "dataset_type": "snapshot",
        "total_items": snapshot_res.total_items,
        "entity_dimension": profile.entity_dimension,
        "primary_metric": profile.primary_metric,
        "pareto_text": snapshot_res.pareto_text,
        "top_item": snapshot_res.top_entities[0].entity if snapshot_res.top_entities else None,
        "top_item_value": snapshot_res.top_entities[0].value if snapshot_res.top_entities else None,
    }
    render_insight_card("Executive", facts_packet, model, allow_paid=allow_paid)


def render_snapshot_breakdown_view(
    df: pd.DataFrame,
    snapshot_res: SnapshotAnalyticsResult,
    profile: DatasetProfile,
) -> None:
    st.header("Rankings & Entity Breakdown")
    st.caption("Deep-dive across all entities, sorted by volume, value, or rate.")

    entity = profile.entity_dimension or df.columns[0]
    metric_options = [m for m in df.columns if pd.api.types.is_numeric_dtype(df[m])]

    if not metric_options:
        st.info("No numeric metrics available for breakdown.")
        return

    sel_metric = st.selectbox(
        "Rank entities by",
        metric_options,
        index=0 if profile.primary_metric not in metric_options else metric_options.index(profile.primary_metric),
        key="snapshot_breakdown_metric_sel",
        format_func=humanize_col,
    )

    h_metric = humanize_col(sel_metric)
    h_entity = humanize_col(entity)

    # Sort descending
    sorted_df = df[[entity, sel_metric]].dropna().sort_values(sel_metric, ascending=False)
    top_15 = sorted_df.head(15)

    fig = px.bar(
        top_15,
        x=sel_metric,
        y=entity,
        orientation="h",
        labels={sel_metric: h_metric, entity: h_entity},
        title=f"Top 15 {h_entity}s by {h_metric}",
        template="plotly_white",
        height=480,
    )
    # Highest on top
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", key="snapshot_breakdown_chart")
    st.session_state.setdefault("_export_figures", {})["Snapshot_breakdown"] = fig

    # Table of all items
    with st.expander("📋 Full Ranking Table", expanded=True):
        st.dataframe(sorted_df.reset_index(drop=True))


def render_snapshot_composition_view(
    df: pd.DataFrame,
    snapshot_res: SnapshotAnalyticsResult,
    profile: DatasetProfile,
) -> None:
    st.header("Composition & Distribution")
    st.caption("Understand portfolio mix, concentration, and relationship between metrics.")

    entity = profile.entity_dimension or df.columns[0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    c1, c2 = st.columns(2)

    with c1:
        if profile.primary_metric and profile.primary_metric in df.columns:
            st.subheader(f"{humanize_col(profile.primary_metric)} Share")
            # Build Top 6 + Others
            sorted_df = df[[entity, profile.primary_metric]].dropna().sort_values(profile.primary_metric, ascending=False)
            top_6 = sorted_df.head(6).copy()
            others_val = sorted_df.iloc[6:][profile.primary_metric].sum() if len(sorted_df) > 6 else 0
            if others_val > 0:
                others_row = pd.DataFrame([{entity: "All Others", profile.primary_metric: others_val}])
                top_6 = pd.concat([top_6, others_row], ignore_index=True)

            fig_donut = px.pie(
                top_6,
                names=entity,
                values=profile.primary_metric,
                hole=0.4,
                title=f"Share of {humanize_col(profile.primary_metric)}",
                template="plotly_white",
            )
            st.plotly_chart(fig_donut, width="stretch", key="snapshot_donut_chart")
            st.session_state.setdefault("_export_figures", {})["Snapshot_composition"] = fig_donut

    with c2:
        if len(num_cols) >= 2:
            st.subheader("Metric Correlation")
            x_m = num_cols[0]
            y_m = num_cols[1] if len(num_cols) > 1 else num_cols[0]
            fig_scatter = px.scatter(
                df,
                x=x_m,
                y=y_m,
                hover_name=entity,
                labels={x_m: humanize_col(x_m), y_m: humanize_col(y_m)},
                title=f"{humanize_col(y_m)} vs {humanize_col(x_m)}",
                template="plotly_white",
            )
            st.plotly_chart(fig_scatter, width="stretch", key="snapshot_scatter_chart")


def render_snapshot_outliers_view(snapshot_res: SnapshotAnalyticsResult) -> None:
    st.header("Risk & Outlier Signals")
    st.caption("Flagged items showing stagnant inventory, zero velocity, or extreme valuation.")

    if not snapshot_res.outliers:
        st.info("✅ No extreme anomalies or dead-stock items detected in this dataset.")
        return

    for o in snapshot_res.outliers:
        severity = o.get("severity", "medium")
        icon = "🔴" if severity == "high" else "🟠"
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"{icon} **{o['item']}** — *{o['issue']}*")
                st.write(o["detail"])
            with col2:
                st.caption(f"Severity: `{severity.upper()}`")
