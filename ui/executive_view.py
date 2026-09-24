"""
ui/executive_view.py — Executive tab: top KPI tiles across the primary +
secondary metrics, plus the primary KPI's trend chart and insight card.
"""

from __future__ import annotations

import streamlit as st

from ui.components import build_trend_figure, render_insight_card, render_kpi_tile


def render_executive_view(pipeline_result: dict, model: str, allow_paid: bool = False) -> None:
    st.header("Executive")

    kpis = pipeline_result["kpis"]
    if not kpis:
        st.info("No numeric metric columns were detected — nothing to summarize here.")
        return

    cols = st.columns(min(4, len(kpis)))
    for i, kpi in enumerate(list(kpis.values())[:4]):
        with cols[i % len(cols)]:
            render_kpi_tile(kpi)

    primary_name = pipeline_result.get("primary_metric")
    trend = pipeline_result.get("trend")

    if trend is not None and primary_name:
        chart_type = st.selectbox(
            "Trend chart type",
            ["Line", "Area"],
            index=0,
            key="exec_trend_chart_type",
            help="Switch between a Line and Area (filled) chart.",
        )
        fig = build_trend_figure(
            trend.series,
            title=primary_name,
            chart_type=chart_type,
        )
        st.plotly_chart(fig, width="stretch", key="executive_trend_chart")
        st.session_state.setdefault("_export_figures", {})["Executive_trend"] = fig
    else:
        if "_export_figures" in st.session_state:
            st.session_state["_export_figures"].pop("Executive_trend", None)
        st.info("No usable date column was found — trend view disabled for this dataset.")

    facts_packet = pipeline_result.get("facts_packets", {}).get(primary_name)
    if facts_packet:
        render_insight_card("Executive", facts_packet, model, allow_paid=allow_paid)
