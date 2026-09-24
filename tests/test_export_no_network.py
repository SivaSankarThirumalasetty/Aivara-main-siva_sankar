import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
import pytest
from pptx import Presentation

from export.pptx_builder import (
    DeckInputs,
    KPITileData,
    ViewExportData,
    build_deck,
    build_risk_signals_slide,
    build_view_slide,
)
from insight import cache as insight_cache


def test_pptx_builder_does_not_import_insight_client():
    """Static check: export/pptx_builder.py's source must never reference
    insight.client, so export can't accidentally start making network calls."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export", "pptx_builder.py"
    )
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert "insight.client" not in imported_modules


def test_export_builds_deck_without_network_call(monkeypatch):
    called = {"hit": False}

    def _fail_if_called(*args, **kwargs):
        called["hit"] = True
        raise AssertionError("OpenRouter client must never be called during export")

    import insight.client as client_module

    monkeypatch.setattr(client_module, "generate_insight", _fail_if_called)
    monkeypatch.setattr(client_module, "_call_openrouter", _fail_if_called)

    insight_cache.set_for_view(
        "Executive",
        {"headline": "Revenue grew 8%.", "driver_explanation": "North led the gain.", "suggested_action": "Review stock."},
    )

    tiles = [KPITileData(label="Revenue", value_display="1.2M", delta_display="+8.0%", delta_positive=True)]
    views = [ViewExportData(view_name="Executive", kpi_tiles=tiles)]
    inputs = DeckInputs(dataset_name="test.csv", period_covered="this month vs last month", views=views, risk_signals=[])

    buffer = build_deck(inputs)
    assert buffer.getbuffer().nbytes > 0
    assert not called["hit"]


def test_export_omits_insight_text_when_not_cached():
    tiles = [KPITileData(label="Orders", value_display="500")]
    views = [ViewExportData(view_name="ViewWithNoInsightGeneratedYet", kpi_tiles=tiles)]
    inputs = DeckInputs(dataset_name="test.csv", period_covered="n/a", views=views, risk_signals=[])

    buffer = build_deck(inputs)
    assert buffer.getbuffer().nbytes > 0


def test_no_risk_slide_when_no_signals():
    prs = Presentation()
    slide_count_before = len(prs.slides)
    build_risk_signals_slide(prs, [])
    assert len(prs.slides) == slide_count_before


def test_export_kaleido_chart_failure_gracefully_degrades(monkeypatch):
    """If kaleido raises an OSError/RuntimeError/Exception during chart export,
    the PPTX build must NOT crash — it simply omits the picture and completes the deck."""
    dummy_fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

    def _broken_figure_to_png_stream(fig, **kwargs):
        raise RuntimeError("Simulated Kaleido process crash / missing library")

    monkeypatch.setattr("export.pptx_builder.figure_to_png_stream", _broken_figure_to_png_stream)

    tiles = [KPITileData(label="Revenue", value_display="1.2M")]
    views = [ViewExportData(view_name="Executive", kpi_tiles=tiles, trend_figure=dummy_fig)]
    inputs = DeckInputs(
        dataset_name="test.csv",
        period_covered="this week vs last week",
        views=views,
        risk_signals=[{"description": "Outlier in North", "severity": "medium"}],
    )

    buffer = build_deck(inputs)
    assert buffer.getbuffer().nbytes > 0
    prs = Presentation(buffer)
    assert len(prs.slides) >= 3


def test_export_multiple_metrics_and_empty_drivers():
    tiles = [
        KPITileData(label="Revenue", value_display="100K", delta_display="+10%"),
        KPITileData(label="Orders", value_display="500", delta_display="+5%"),
        KPITileData(label="AOV", value_display="$200", delta_display="+4.7%"),
        KPITileData(label="Margin", value_display="35%", delta_display="+2.1%"),
    ]
    views = [ViewExportData(view_name="Executive", kpi_tiles=tiles, drivers_table=[], drivers_figure=None)]
    inputs = DeckInputs(dataset_name="multi.csv", period_covered="n/a", views=views, risk_signals=[])
    buf = build_deck(inputs)
    assert buf.getbuffer().nbytes > 0
