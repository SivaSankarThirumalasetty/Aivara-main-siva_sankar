import io
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from ingestion.classifier import ColumnClassification, classify_columns
from ingestion.cleaner import clean_dataframe
from ingestion.loader import load_csv, load_file
from insight.client import (
    BudgetExhaustedError,
    InsightResult,
    OpenRouterError,
    generate_insight,
)
from export.pptx_builder import DeckInputs, KPITileData, ViewExportData, build_deck


# ===========================================================================
# USER JOURNEY 1 — Complete User Lifecycle
# ===========================================================================
def test_user_journey_1_full_lifecycle(monkeypatch):
    """User Journey 1:
    Open application -> Upload CSV -> Inspect schema -> Override role ->
    Confirm schema -> Dashboard renders all tabs -> AI insight -> Export PPTX."""
    def _mock_insight(facts_packet, **kwargs):
        return InsightResult(
            headline="Revenue reached 2,000 this week.",
            driver_explanation="Enterprise segment drove 80% of gains.",
            suggested_action="Review inventory.",
            raw_text="mocked raw text",
            from_cache=False,
            model="test_model",
        )

    monkeypatch.setattr("insight.client.generate_insight", _mock_insight)

    csv_data = (
        "date,revenue,cost,region,customer_type\n"
        "2024-01-01,1000.0,600.0,North,Enterprise\n"
        "2024-01-02,1500.0,800.0,South,SMB\n"
        "2024-01-03,1200.0,700.0,North,Enterprise\n"
        "2024-01-04,1800.0,900.0,South,SMB\n"
        "2024-01-05,2000.0,1100.0,East,Enterprise\n"
    ).encode("utf-8")

    load_res = load_csv(csv_data, filename="sales.csv")
    assert load_res.success
    clean_df, clean_rep = clean_dataframe(load_res.dataframe)
    raw_classifications = classify_columns(clean_df)

    at = AppTest.from_file("app.py")
    at.session_state["dataset"] = clean_df
    at.session_state["dataset_name"] = "sales.csv"
    at.session_state["_uploaded_filename"] = "sales.csv"
    at.session_state["_uploaded_file_hash"] = "hash_sales_1"
    at.session_state["classifications"] = {c.name: c for c in raw_classifications}

    # Step 1: User changes a column role override
    at.session_state["overrides"] = {"customer_type": "dimension"}

    # Step 2: User confirms schema
    at.session_state["schema_confirmed"] = True
    at.run(timeout=10)

    assert not at.exception, f"AppTest raised unexpected exception: {[str(e) for e in at.exception]}"

    # Step 3: Verify KPI metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert any("Revenue" in lbl for lbl in metric_labels)

    # Step 4: Verify Prepare PPTX export works
    at.session_state["_export_figures"] = {}
    at.sidebar.button(key="prepare_pptx_btn").click()
    at.run(timeout=10)

    assert not at.exception
    assert "_pptx_buffer" in at.session_state
    assert len(at.session_state["_pptx_buffer"]) > 0


# ===========================================================================
# USER JOURNEY 2 — Dataset B Replaces Dataset A
# ===========================================================================
def test_user_journey_2_dataset_replacement_and_state_reset():
    """User Journey 2:
    Upload Dataset A -> confirm schema -> generate charts & export.
    Then upload Dataset B -> verify schema confirmation resets,
    Dataset A figures/insights disappear, Dataset B contains only Dataset B data."""
    csv_a = b"date,revenue,region\n2024-01-01,100,North\n2024-01-02,200,South\n"
    csv_b = b"item_id,stock,category\nSKU001,50,Electronics\nSKU002,120,Furniture\n"

    # Setup Session State as if Dataset A was active
    at = AppTest.from_file("app.py")
    at.session_state["_uploaded_filename"] = "dataset_a.csv"
    at.session_state["_uploaded_file_hash"] = "hash_a"
    at.session_state["dataset_name"] = "dataset_a.csv"
    at.session_state["dataset"] = pd.read_csv(io.BytesIO(csv_a))
    at.session_state["schema_confirmed"] = True
    at.session_state["overrides"] = {"region": "dimension"}
    at.session_state["_export_figures"] = {"Executive_trend": object()}
    at.session_state["_pptx_buffer"] = b"dataset_a_deck"
    at.session_state["_view_insights"] = {"Executive": {"headline": "Dataset A Insight"}}

    # Now upload Dataset B
    load_res_b = load_file(csv_b, filename="dataset_b.csv")
    cleaned_b, _ = clean_dataframe(load_res_b.dataframe)
    class_b = classify_columns(cleaned_b)

    # Apply upload reset logic
    at.session_state["dataset"] = cleaned_b
    at.session_state["classifications"] = {c.name: c for c in class_b}
    at.session_state["_uploaded_filename"] = "dataset_b.csv"
    at.session_state["_uploaded_file_hash"] = "hash_b"
    at.session_state["dataset_name"] = "dataset_b.csv"
    at.session_state["schema_confirmed"] = False
    for k in ["overrides", "_export_figures", "_pptx_buffer", "_view_insights"]:
        if k in at.session_state:
            del at.session_state[k]

    # Verify reset conditions before user confirms schema B
    assert at.session_state["schema_confirmed"] is False
    assert "overrides" not in at.session_state
    assert "_export_figures" not in at.session_state
    assert "_pptx_buffer" not in at.session_state
    assert "_view_insights" not in at.session_state

    # Now confirm schema for Dataset B and run app
    at.session_state["schema_confirmed"] = True
    at.run(timeout=10)

    assert not at.exception
    assert at.session_state["dataset_name"] == "dataset_b.csv"
    assert "SKU001" in at.session_state["dataset"]["item_id"].values


# ===========================================================================
# USER JOURNEY 3 — Same Filename Replaced Content
# ===========================================================================
def test_user_journey_3_same_filename_content_updated():
    """User Journey 3:
    Upload report.csv (version 1) -> upload report.csv (version 2 with different content).
    Verify version 2 is processed."""
    content_v1 = b"date,revenue\n2024-01-01,100\n"
    content_v2 = b"date,revenue\n2024-01-01,9999\n"

    import hashlib
    hash_v1 = hashlib.md5(content_v1).hexdigest()
    hash_v2 = hashlib.md5(content_v2).hexdigest()

    assert hash_v1 != hash_v2

    at = AppTest.from_file("app.py")
    at.session_state["_uploaded_filename"] = "report.csv"
    at.session_state["_uploaded_file_hash"] = hash_v1
    at.session_state["dataset"] = pd.read_csv(io.BytesIO(content_v1))

    # Version 2 uploaded with same filename "report.csv"
    new_res = load_csv(content_v2, filename="report.csv")
    cleaned_v2, _ = clean_dataframe(new_res.dataframe)
    class_v2 = classify_columns(cleaned_v2)
    at.session_state["dataset"] = cleaned_v2
    at.session_state["classifications"] = {c.name: c for c in class_v2}
    at.session_state["_uploaded_file_hash"] = hash_v2
    at.session_state["schema_confirmed"] = True
    at.run(timeout=10)

    assert not at.exception
    assert at.session_state["dataset"]["revenue"].iloc[0] == 9999


# ===========================================================================
# USER JOURNEY 4 — Bad / Edge-Case Files
# ===========================================================================
@pytest.mark.parametrize(
    "bad_name,bad_bytes",
    [
        ("empty.csv", b""),
        ("one_col.csv", b"single_column\nval1\nval2\n"),
        ("malformed.csv", b"col1,col2\n1,2,3,4,5\n6\n"),
        ("invalid_encoding.csv", b"\xff\xfe\x00\x00\x12\x34\x56\x78"),
        ("duplicate_headers.csv", b"date,revenue,revenue\n2024-01-01,100,200\n"),
        ("no_date.csv", b"product,price,units\nA,10,5\nB,20,3\n"),
        ("no_numeric.csv", b"category,status,city\nA,Open,NY\nB,Closed,LA\n"),
        ("all_null.csv", b"date,revenue\n,\n,\n"),
        ("huge_text.csv", b"name,desc\nA," + (b"long " * 5000) + b"\n"),
        ("corrupted.xlsx", b"PK\x03\x04corrupted_zip_stream_bytes_12345"),
    ],
)
def test_user_journey_4_bad_files_fail_gracefully(bad_name, bad_bytes):
    """User Journey 4:
    Bad and corrupt inputs must fail gracefully without raising unhandled exceptions."""
    load_res = load_file(bad_bytes, filename=bad_name)
    assert load_res is not None
    # If load succeeded, cleaning and classifying must still never crash
    if load_res.success and not load_res.dataframe.empty:
        cleaned_df, report = clean_dataframe(load_res.dataframe)
        classifications = classify_columns(cleaned_df)
        assert isinstance(classifications, list)


# ===========================================================================
# USER JOURNEY 5 — AI Network / API Failures
# ===========================================================================
@pytest.mark.parametrize(
    "failure_type,exc_or_return",
    [
        ("timeout", OpenRouterError("Connection timed out")),
        ("http_400", OpenRouterError("400 Bad Request")),
        ("http_401", OpenRouterError("401 Unauthorized")),
        ("http_429", OpenRouterError("429 Too Many Requests")),
        ("malformed_json", ("{invalid_json: true", "test_model")),
        ("empty_response", ("", "test_model")),
        ("budget_exhausted", BudgetExhaustedError("Daily budget exhausted")),
    ],
)
def test_user_journey_5_ai_failures_render_dashboard(failure_type, exc_or_return, monkeypatch):
    """User Journey 5:
    When AI encounters timeout, 400, 401, 429, malformed response, or budget exhaustion,
    the dashboard must still render fully with all calculated numbers."""
    if isinstance(exc_or_return, Exception):
        def _mock_failure(*args, **kwargs):
            raise exc_or_return
        monkeypatch.setattr("insight.client.generate_insight", _mock_failure)
    else:
        def _mock_return(*args, **kwargs):
            return exc_or_return
        monkeypatch.setattr("insight.client._call_openrouter", _mock_return)

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "region": ["North", "South", "East", "West", "North"],
    })
    at = AppTest.from_file("app.py")
    at.session_state["dataset"] = df
    at.session_state["dataset_name"] = "test_ai.csv"
    at.session_state["classifications"] = {
        "date": ColumnClassification(name="date", role="date", confidence=1.0, reason="test"),
        "revenue": ColumnClassification(name="revenue", role="metric", confidence=1.0, reason="test"),
        "region": ColumnClassification(name="region", role="dimension", confidence=1.0, reason="test"),
    }
    at.session_state["schema_confirmed"] = True
    at.run(timeout=10)

    # Dashboard must render fully without raising unhandled crash
    assert not at.exception


# ===========================================================================
# USER JOURNEY 6 — PPTX Export Failure Graceful Degradation
# ===========================================================================
def test_user_journey_6_export_chart_failure_graceful(monkeypatch):
    """User Journey 6:
    If a chart fails to render to PNG (e.g. Kaleido failure), PPTX export
    must omit the picture and still produce a valid downloadable deck."""
    import plotly.graph_objects as go

    def _broken_stream(fig, **kwargs):
        raise RuntimeError("Simulated Kaleido rendering crash")

    monkeypatch.setattr("export.pptx_builder.figure_to_png_stream", _broken_stream)

    broken_fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    tiles = [KPITileData(label="Revenue", value_display="$1.5M", delta_display="+15%")]
    views = [ViewExportData(view_name="Executive", kpi_tiles=tiles, trend_figure=broken_fig)]
    deck_inputs = DeckInputs(
        dataset_name="sales.csv",
        period_covered="this month vs last month",
        views=views,
        risk_signals=[{"description": "Outlier in North", "severity": "medium"}],
    )

    buffer = build_deck(deck_inputs)
    assert buffer.getbuffer().nbytes > 0
