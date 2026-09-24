"""
ui/revenue_view.py — a generalized "primary metric" view (not literally
revenue): deep-dive on the primary KPI with its trend and driver breakdown.
"""

from __future__ import annotations

import streamlit as st

from ui.components import build_drivers_figure, build_trend_figure, render_insight_card, render_kpi_tile


def render_primary_metric_view(pipeline_result: dict, model: str, allow_paid: bool = False) -> None:
    primary_name = pipeline_result.get("primary_metric")
    st.header(primary_name.replace("_", " ").title() if primary_name else "Primary metric")

    kpis = pipeline_result["kpis"]
    if not primary_name or primary_name not in kpis:
        st.info("No primary metric available for this dataset.")
        return

    render_kpi_tile(kpis[primary_name])

    trend = pipeline_result.get("trend")
    if trend is not None:
        chart_type = st.selectbox(
            "Trend chart type",
            ["Line", "Area", "Bar"],
            index=0,
            key="revenue_trend_chart_type",
            help="Switch chart style for the trend visualization.",
        )
        if chart_type == "Bar":
            import plotly.express as px
            from ui.components import humanize_col
            h_name = humanize_col(primary_name)
            fig = px.bar(
                trend.series,
                x="period",
                y="value",
                labels={"period": "Period", "value": h_name},
                title=h_name,
                template="plotly_white",
                height=420,
            )
        else:
            fig = build_trend_figure(
                trend.series,
                title=primary_name,
                chart_type=chart_type,
            )
        st.plotly_chart(fig, width="stretch", key="revenue_trend_chart")
        st.session_state.setdefault("_export_figures", {})["Revenue_trend"] = fig
    else:
        if "_export_figures" in st.session_state:
            st.session_state["_export_figures"].pop("Revenue_trend", None)

    top_driver = pipeline_result.get("top_dimension_driver")
    if top_driver is not None and top_driver.contributions:
        fig = build_drivers_figure(
            top_driver.top_positive + top_driver.top_negative,
            top_driver.dimension,
        )
        st.plotly_chart(fig, width="stretch", key="revenue_drivers_chart")
        st.session_state.setdefault("_export_figures", {})["Revenue_drivers"] = fig
    else:
        if "_export_figures" in st.session_state:
            st.session_state["_export_figures"].pop("Revenue_drivers", None)
        st.info("No dimension columns with enough contrast were found to explain the change.")

    facts_packet = pipeline_result.get("facts_packets", {}).get(primary_name)
    if facts_packet:
        render_insight_card("Revenue", facts_packet, model, allow_paid=allow_paid)
