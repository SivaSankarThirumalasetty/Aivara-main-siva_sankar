"""
ui/components.py — shared widgets: KPI tile, insight card, chart wrappers.

Plotly figures built here are the same figure objects reused later for PPTX
export (export/chart_images.py) — we never rebuild a chart with a different
library for the deck.
"""

from __future__ import annotations

import math
import re
import streamlit as st
import plotly.graph_objects as go

from analytics.kpis import KPIResult
from insight.client import BudgetExhaustedError, InsightResult


# ---------------------------------------------------------------------------
# Column-name humanizer — converts snake_case, camelCase, ALL_CAPS, and
# already-spaced Excel headers into readable "Title Case" labels that appear
# on chart axes and titles instead of raw internal column names.
# ---------------------------------------------------------------------------

_WORD_SPLIT_RE = re.compile(
    r"""
    (?<=[a-z])(?=[A-Z]) |   # camelCase boundary: lowercase→Uppercase
    (?<=[A-Z])(?=[A-Z][a-z]) |  # ACRONYMWord boundary: URL→Pars → URL, Pars
    [\s_\-]+                # explicit separator (underscore, dash, space)
    """,
    re.VERBOSE,
)


def humanize_col(col: str) -> str:
    """Convert a raw column name into a human-readable label.

    Examples
    --------
    >>> humanize_col("total_revenue")  ->  "Total Revenue"
    >>> humanize_col("salesAmount")    ->  "Sales Amount"
    >>> humanize_col("GROSS_MARGIN")   ->  "Gross Margin"
    >>> humanize_col("Revenue")        ->  "Revenue"
    >>> humanize_col("customer_id")    ->  "Customer Id"
    """
    words = [w for w in _WORD_SPLIT_RE.split(str(col)) if w]
    return " ".join(w.capitalize() for w in words)


def format_number(value: float, unit: str | None = None) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    abs_value = abs(value)
    if abs_value >= 1_00_00_000:  # 1 crore, in case of INR-style data
        return f"{value / 1_00_00_000:.2f}Cr"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}" if isinstance(value, float) and not value.is_integer() else f"{value:,.0f}"


def format_pct_change(
    pct_change: float | None, with_arrow: bool = False, with_sign: bool = False
) -> str | None:
    if pct_change is None:
        return None
    if math.isinf(pct_change):
        return "new" if pct_change > 0 else "n/a (no prior period)"
    if with_arrow:
        arrow = "↑" if pct_change >= 0 else "↓"
        return f"{arrow} {abs(pct_change):.1f}%"
    if with_sign:
        return f"{pct_change:+.1f}%"
    return f"{pct_change:.1f}%"


def render_kpi_tile(kpi: KPIResult, unit: str | None = None) -> None:
    display_unit = unit or kpi.unit
    value_str = format_number(kpi.current_value, display_unit)
    delta_str = format_pct_change(kpi.pct_change, with_arrow=True)

    st.metric(label=kpi.name.replace("_", " ").title(), value=value_str, delta=delta_str)


def render_insight_card(
    view_name: str, facts_packet: dict, model: str, allow_paid: bool = False
) -> InsightResult | None:
    """Attempts to generate (or fetch cached) an insight for this view and
    renders it. On budget exhaustion or API failure, renders a neutral
    placeholder instead of erroring — the dashboard must always render fully
    even if the LLM is unavailable (Section 9.2.7)."""
    from insight import cache as insight_cache
    from insight.client import OpenRouterError, generate_insight

    try:
        result = generate_insight(facts_packet, model=model, allow_paid=allow_paid)
        insight_cache.set_for_view(
            view_name,
            {
                "headline": result.headline,
                "driver_explanation": result.driver_explanation,
                "suggested_action": result.suggested_action,
                "raw_text": result.raw_text,
                "model": result.model,
            },
        )
        with st.container(border=True):
            st.markdown(f"**{result.headline}**")
            if result.driver_explanation:
                st.write(result.driver_explanation)
            if result.suggested_action:
                st.markdown(f"*Next: {result.suggested_action}*")
            if result.model != model:
                st.caption(f"ℹ️ Used fallback model `{result.model}` (free tier)")
        return result
    except BudgetExhaustedError:
        st.info("Insight budget reached for today — showing computed numbers only.")
    except OpenRouterError as exc:
        st.info("Insight temporarily unavailable — showing computed numbers only.")
        st.caption(f"(details: {exc})")
    return None


# ---------------------------------------------------------------------------
# Chart builders — all figures use humanized column names as axis labels.
# ---------------------------------------------------------------------------

_TREND_CHART_TYPES = ["Line", "Area"]
_TREND_COLORS = {"Line": "#1a73e8", "Area": "#1a73e8"}


def build_trend_figure(
    trend_df,
    title: str,
    chart_type: str = "Line",
    x_col: str = "period",
    y_col: str = "value",
    y_label: str | None = None,
) -> go.Figure:
    """Build a trend chart (Line or Area) with humanized axis labels.

    Parameters
    ----------
    trend_df : DataFrame with `period` and `value` columns (TrendResult.series).
    title    : Raw metric column name — will be humanized for the chart title.
    chart_type : "Line" or "Area".
    y_label  : Override for Y-axis label; defaults to humanized `title`.
    """
    h_title = humanize_col(title)
    h_ylabel = y_label if y_label is not None else h_title
    color = _TREND_COLORS.get(chart_type, "#1a73e8")

    fig = go.Figure()
    if chart_type == "Area":
        fig.add_trace(
            go.Scatter(
                x=trend_df[x_col],
                y=trend_df[y_col],
                mode="lines+markers",
                name=h_title,
                fill="tozeroy",
                line=dict(color=color),
                fillcolor=f"rgba(26,115,232,0.15)",
            )
        )
    else:  # Line (default)
        fig.add_trace(
            go.Scatter(
                x=trend_df[x_col],
                y=trend_df[y_col],
                mode="lines+markers",
                name=h_title,
                line=dict(color=color),
            )
        )

    fig.update_layout(
        title=h_title,
        xaxis_title="Period",
        yaxis_title=h_ylabel,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        height=420,
    )
    return fig


def build_drivers_figure(
    contributions,
    dimension_name: str,
    x_label: str | None = None,
    y_label: str | None = None,
) -> go.Figure:
    """Build a waterfall-style bar chart showing contribution % by category.

    Uses humanized column names for the dimension axis and Y-axis title.
    """
    h_dimension = humanize_col(dimension_name)
    h_ylabel = y_label if y_label is not None else "Contribution to Change (%)"
    h_xlabel = x_label if x_label is not None else h_dimension

    categories = [c.category for c in contributions]
    deltas = [c.contribution_pct for c in contributions]
    colors = ["#1e7a34" if d >= 0 else "#b02a2a" for d in deltas]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=deltas,
            marker_color=colors,
            text=[f"{d:+.1f}%" for d in deltas],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Contribution to Change by {h_dimension}",
        xaxis_title=h_xlabel,
        yaxis_title=h_ylabel,
        margin=dict(l=40, r=20, t=50, b=60),
        template="plotly_white",
        height=420,
    )
    return fig


def mask_secret(secret: str, keep: int = 4) -> str:
    if not secret:
        return ""
    if len(secret) <= keep:
        return "*" * len(secret)
    return "*" * (len(secret) - keep) + secret[-keep:]
