import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go

from ui.components import humanize_col, build_trend_figure, build_drivers_figure
from ui.chart_builder import build_chart_figure, CHART_TYPES


# ---------------------------------------------------------------------------
# humanize_col tests
# ---------------------------------------------------------------------------

def test_humanize_snake_case():
    assert humanize_col("total_revenue") == "Total Revenue"


def test_humanize_camel_case():
    assert humanize_col("salesAmount") == "Sales Amount"


def test_humanize_all_caps_underscore():
    assert humanize_col("GROSS_MARGIN") == "Gross Margin"


def test_humanize_already_title():
    assert humanize_col("Revenue") == "Revenue"


def test_humanize_with_spaces():
    # Excel headers may already have spaces
    assert humanize_col("Customer Segment") == "Customer Segment"


def test_humanize_single_word():
    assert humanize_col("sales") == "Sales"


# ---------------------------------------------------------------------------
# build_trend_figure tests
# ---------------------------------------------------------------------------

def _sample_trend_df():
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "value": [100.0, 120.0, 90.0, 140.0, 130.0, 160.0],
    })


def test_trend_figure_line_returns_figure():
    fig = build_trend_figure(_sample_trend_df(), title="total_revenue", chart_type="Line")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_trend_figure_area_fills():
    fig = build_trend_figure(_sample_trend_df(), title="sales_amount", chart_type="Area")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].fill == "tozeroy"


def test_trend_figure_uses_humanized_title():
    fig = build_trend_figure(_sample_trend_df(), title="gross_margin", chart_type="Line")
    assert fig.layout.title.text == "Gross Margin"


def test_trend_figure_has_axis_labels():
    fig = build_trend_figure(_sample_trend_df(), title="revenue", chart_type="Line")
    assert fig.layout.yaxis.title.text == "Revenue"
    assert fig.layout.xaxis.title.text == "Period"


# ---------------------------------------------------------------------------
# build_drivers_figure tests
# ---------------------------------------------------------------------------

class _MockContribution:
    def __init__(self, category, contribution_pct):
        self.category = category
        self.contribution_pct = contribution_pct


def test_drivers_figure_returns_figure():
    contributions = [
        _MockContribution("North", 45.0),
        _MockContribution("South", -25.0),
    ]
    fig = build_drivers_figure(contributions, "region")
    assert isinstance(fig, go.Figure)


def test_drivers_figure_humanized_title():
    contributions = [_MockContribution("A", 10.0)]
    fig = build_drivers_figure(contributions, "product_category")
    assert "Product Category" in fig.layout.title.text


def test_drivers_figure_has_axis_labels():
    contributions = [_MockContribution("A", 10.0)]
    fig = build_drivers_figure(contributions, "customer_segment")
    assert fig.layout.yaxis.title.text == "Contribution to Change (%)"
    assert "Customer Segment" in fig.layout.xaxis.title.text


# ---------------------------------------------------------------------------
# build_chart_figure tests — one for each chart type
# ---------------------------------------------------------------------------

def _make_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "revenue": [float(i * 10 + 50) for i in range(20)],
        "cost": [float(i * 5 + 20) for i in range(20)],
        "region": (["North", "South"] * 10),
        "category": (["A", "B", "C", "D"] * 5),
    })


def test_chart_line():
    df = _make_df()
    fig = build_chart_figure(df, "Line", x_col="date", y_col="revenue")
    assert isinstance(fig, go.Figure)


def test_chart_area():
    df = _make_df()
    fig = build_chart_figure(df, "Area", x_col="date", y_col="revenue")
    assert isinstance(fig, go.Figure)
    # area fill should be set on trace
    assert fig.data[0].fill is not None


def test_chart_bar():
    df = _make_df()
    fig = build_chart_figure(df, "Bar", x_col="region", y_col="revenue")
    assert isinstance(fig, go.Figure)


def test_chart_horizontal_bar():
    df = _make_df()
    fig = build_chart_figure(df, "Horizontal Bar", x_col="region", y_col="revenue")
    assert isinstance(fig, go.Figure)


def test_chart_scatter():
    df = _make_df()
    fig = build_chart_figure(df, "Scatter", x_col="cost", y_col="revenue")
    assert isinstance(fig, go.Figure)


def test_chart_pie():
    df = _make_df()
    fig = build_chart_figure(df, "Pie", x_col="region", y_col="revenue")
    assert isinstance(fig, go.Figure)


def test_chart_donut():
    df = _make_df()
    fig = build_chart_figure(df, "Donut", x_col="region", y_col="revenue")
    assert isinstance(fig, go.Figure)
    # donut hole should be set
    assert fig.data[0].hole > 0


def test_chart_box():
    df = _make_df()
    fig = build_chart_figure(df, "Box", x_col="region", y_col="revenue")
    assert isinstance(fig, go.Figure)


def test_chart_histogram():
    df = _make_df()
    fig = build_chart_figure(df, "Histogram", x_col="revenue", y_col=None)
    assert isinstance(fig, go.Figure)


def test_chart_heatmap():
    df = _make_df()
    fig = build_chart_figure(df, "Heatmap", x_col=None, y_col=None)
    assert isinstance(fig, go.Figure)


def test_chart_timeline_gantt():
    df = pd.DataFrame({
        "task": ["T1", "T2"],
        "start": pd.to_datetime(["2024-01-01", "2024-01-05"]),
        "end": pd.to_datetime(["2024-01-05", "2024-01-10"]),
    })
    fig = build_chart_figure(df, "Timeline (Gantt)", x_col="start", y_col="end", color_col="task")
    assert isinstance(fig, go.Figure)


def test_chart_empty_df_returns_none():
    fig = build_chart_figure(pd.DataFrame(), "Line", x_col="a", y_col="b")
    assert fig is None


def test_chart_with_color_col():
    df = _make_df()
    fig = build_chart_figure(df, "Bar", x_col="region", y_col="revenue", color_col="category")
    assert isinstance(fig, go.Figure)


def test_chart_uses_humanized_axis_labels():
    df = _make_df()
    fig = build_chart_figure(df, "Bar", x_col="region", y_col="revenue")
    # plotly express sets labels on axes
    assert fig.layout.xaxis.title.text == "Region"
    assert fig.layout.yaxis.title.text == "Revenue"


def test_all_chart_types_covered():
    """Ensure CHART_TYPES list matches what we test above (no new type silently untested)."""
    expected = {
        "Line", "Area", "Bar", "Horizontal Bar", "Scatter",
        "Pie", "Donut", "Box", "Histogram", "Heatmap", "Timeline (Gantt)",
    }
    assert set(CHART_TYPES) == expected
