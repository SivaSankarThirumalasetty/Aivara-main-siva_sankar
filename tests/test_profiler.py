import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from ingestion.classifier import ColumnClassification, ROLE_DATE, ROLE_DIMENSION, ROLE_METRIC
from analytics.profiler import (
    PROFILE_TIMESERIES,
    PROFILE_PROJECT,
    PROFILE_SNAPSHOT,
    profile_dataset,
)


def test_profile_snapshot_inventory():
    df = pd.DataFrame(
        {
            "Product Name": [f"Item {i}" for i in range(20)],
            "Category": (["Electronics", "Office"] * 10),
            "Units Sold": range(20),
            "Total Cost": [i * 100 for i in range(20)],
        }
    )
    classifications = [
        ColumnClassification("Product Name", ROLE_DIMENSION, 1.0, "test"),
        ColumnClassification("Category", ROLE_DIMENSION, 1.0, "test"),
        ColumnClassification("Units Sold", ROLE_METRIC, 1.0, "test"),
        ColumnClassification("Total Cost", ROLE_METRIC, 1.0, "test"),
    ]
    profile = profile_dataset(df, classifications)
    assert profile.profile_type == PROFILE_SNAPSHOT
    assert profile.entity_dimension == "Product Name"
    assert "Category" in profile.grouping_dimensions
    assert profile.primary_metric is not None


def test_profile_project():
    df = pd.DataFrame(
        {
            "Task Name": [f"Task {i}" for i in range(10)],
            "Project Name": (["Alpha", "Beta"] * 5),
            "Start Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "End Date": pd.date_range("2024-01-10", periods=10, freq="D"),
            "Days Required": [10] * 10,
            "Progress": [0.5] * 10,
        }
    )
    classifications = [
        ColumnClassification("Task Name", ROLE_DIMENSION, 1.0, "test"),
        ColumnClassification("Project Name", ROLE_DIMENSION, 1.0, "test"),
        ColumnClassification("Start Date", ROLE_DATE, 1.0, "test"),
        ColumnClassification("End Date", ROLE_DATE, 1.0, "test"),
        ColumnClassification("Days Required", ROLE_METRIC, 1.0, "test"),
        ColumnClassification("Progress", ROLE_METRIC, 1.0, "test", is_rate=True),
    ]
    profile = profile_dataset(df, classifications)
    assert profile.profile_type == PROFILE_PROJECT
    assert profile.primary_date_col == "Start Date"
    assert profile.end_date_col == "End Date"
    assert profile.has_progress is True
    assert profile.duration_metric == "Days Required"


def test_profile_timeseries():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "region": (["North", "South"] * 5),
            "revenue": [1000 + i * 50 for i in range(10)],
        }
    )
    classifications = [
        ColumnClassification("date", ROLE_DATE, 1.0, "test"),
        ColumnClassification("region", ROLE_DIMENSION, 1.0, "test"),
        ColumnClassification("revenue", ROLE_METRIC, 1.0, "test"),
    ]
    profile = profile_dataset(df, classifications)
    assert profile.profile_type == PROFILE_TIMESERIES
    assert profile.primary_date_col == "date"
    assert profile.end_date_col is None
