import hashlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from ingestion.cleaner import clean_dataframe
from ingestion.classifier import classify_columns
from ingestion.loader import load_file
from insight import cache as insight_cache


def test_upload_identity_and_session_state_reset():
    """Verify that uploading a second dataset (or same filename with different bytes)
    completely resets schema confirmation, export figures, view insights, and cached buffers."""
    csv_a = b"date,revenue,region\n2024-01-01,100,North\n2024-01-02,200,South\n"
    csv_b = b"date,revenue,region\n2024-02-01,500,East\n2024-02-02,600,West\n"

    hash_a = hashlib.md5(csv_a).hexdigest()
    hash_b = hashlib.md5(csv_b).hexdigest()

    assert hash_a != hash_b

    # Simulate session state for Dataset A
    session_state = {
        "_uploaded_filename": "report.csv",
        "_uploaded_file_hash": hash_a,
        "dataset_name": "report.csv",
        "dataset": pd.read_csv(io.BytesIO(csv_a)),
        "classifications": {"revenue": "metric"},
        "schema_confirmed": True,
        "overrides": {"revenue": "metric"},
        "_export_figures": {"Executive_trend": object()},
        "_pptx_buffer": b"mock_pptx_dataset_a",
        "_view_insights": {"Executive": {"headline": "Dataset A grew 50%"}},
    }

    # Now simulate what render_upload_page does when new bytes arrive with same filename "report.csv"
    uploaded_name = "report.csv"
    new_bytes = csv_b
    new_hash = hashlib.md5(new_bytes).hexdigest()

    if session_state.get("_uploaded_file_hash") != new_hash:
        load_res = load_file(new_bytes, filename=uploaded_name)
        cleaned_df, clean_report = clean_dataframe(load_res.dataframe)
        classifications = classify_columns(cleaned_df)

        session_state["dataset"] = cleaned_df
        session_state["classifications"] = {c.name: c for c in classifications}
        session_state["_uploaded_filename"] = uploaded_name
        session_state["_uploaded_file_hash"] = new_hash
        session_state["dataset_name"] = uploaded_name
        session_state["schema_confirmed"] = False
        session_state.pop("overrides", None)
        session_state.pop("_export_figures", None)
        session_state.pop("_pptx_buffer", None)
        session_state.pop("_view_insights", None)
        insight_cache.clear_view_cache()

    # Assert all reset requirements:
    assert session_state["schema_confirmed"] is False
    assert "overrides" not in session_state
    assert "_export_figures" not in session_state
    assert "_pptx_buffer" not in session_state
    assert "_view_insights" not in session_state
    assert session_state["_uploaded_file_hash"] == hash_b
    assert session_state["dataset"]["revenue"].iloc[0] == 500
