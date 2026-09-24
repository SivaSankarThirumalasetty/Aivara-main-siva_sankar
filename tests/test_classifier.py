import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ingestion.classifier import ROLE_DATE, ROLE_DIMENSION, ROLE_ID, ROLE_METRIC, classify_columns


def _role_map(df: pd.DataFrame) -> dict:
    return {c.name: c.role for c in classify_columns(df)}


def test_date_column_detected():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 10),
            "value": range(30),
        }
    )
    roles = _role_map(df)
    assert roles["date"] == ROLE_DATE


def test_string_date_column_detected():
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 10,
            "amount": range(30),
        }
    )
    roles = _role_map(df)
    assert roles["order_date"] == ROLE_DATE


def test_metric_column_detected():
    df = pd.DataFrame({"id": range(100), "revenue": [i * 1.5 for i in range(100)]})
    roles = _role_map(df)
    assert roles["revenue"] == ROLE_METRIC


def test_dimension_column_detected():
    df = pd.DataFrame(
        {
            "region": (["North", "South", "East", "West"] * 25),
            "value": range(100),
        }
    )
    roles = _role_map(df)
    assert roles["region"] == ROLE_DIMENSION


def test_id_column_detected():
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i}" for i in range(200)],
            "value": range(200),
        }
    )
    roles = _role_map(df)
    assert roles["customer_id"] == ROLE_ID


def test_numeric_low_cardinality_code_treated_as_dimension():
    df = pd.DataFrame(
        {
            "store_code": ([1, 2, 3, 4, 5] * 40),
            "value": range(200),
        }
    )
    roles = _role_map(df)
    assert roles["store_code"] == ROLE_DIMENSION


def test_entity_name_classified_as_dimension_not_id():
    # 46 products with 46 unique names must be classified as dimension, not ID
    df = pd.DataFrame(
        {
            "product_id": [f"P{100+i}" for i in range(46)],
            "product_name": [f"Product Item {i}" for i in range(46)],
            "revenue": [100.0 + i for i in range(46)],
        }
    )
    classified = classify_columns(df)
    cmap = {c.name: c for c in classified}
    assert cmap["product_id"].role == ROLE_ID
    assert cmap["product_name"].role == ROLE_DIMENSION


def test_progress_rate_metric_detected():
    df = pd.DataFrame(
        {
            "task": [f"Task {i}" for i in range(10)],
            "progress": [0.0, 0.25, 0.5, 0.75, 1.0, 0.33, 0.8, 0.45, 0.9, 0.1],
        }
    )
    classified = classify_columns(df)
    cmap = {c.name: c for c in classified}
    assert cmap["progress"].role == ROLE_METRIC
    assert cmap["progress"].is_rate is True
