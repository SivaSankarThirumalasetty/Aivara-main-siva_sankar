import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.drivers import compute_dimension_drivers
from analytics.kpis import compute_all_kpis, compute_kpi
from analytics.trend import build_trend_series
from ingestion.classifier import ROLE_DATE, classify_columns
from ingestion.cleaner import clean_dataframe
from ui.components import format_pct_change


def test_scenario_1_revenue_additive_sum():
    """TEST 1 — Revenue: Jan 1: 100, Jan 2: 200, Jan 3: 300. Expected total = 600."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Revenue": [100.0, 200.0, 300.0],
    })
    res = compute_kpi(df, None, "Revenue")
    assert res.current_value == 600.0


def test_scenario_2_conversion_rate_simple_mean():
    """TEST 2 — Conversion rate: Jan 1: 10%, Jan 2: 20%, Jan 3: 30%. Expected mean = 20%, NOT 60%."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Conversion_Rate": [0.10, 0.20, 0.30],
    })
    res = compute_kpi(df, None, "Conversion_Rate")
    assert abs(res.current_value - 0.20) < 1e-6
    assert res.current_value != 0.60


def test_scenario_3_offset_drivers():
    """TEST 3 — Offset drivers: North Prior=1000, Current=2000; South Prior=2000, Current=1000.
    Net change = 0. Shows North +1000 (+50%) and South -1000 (-50%)."""
    df_prior = pd.DataFrame({"Region": ["North", "South"], "Revenue": [1000.0, 2000.0]})
    df_current = pd.DataFrame({"Region": ["North", "South"], "Revenue": [2000.0, 1000.0]})

    res = compute_dimension_drivers(df_current, df_prior, "Region", "Revenue")
    assert res is not None
    assert abs(res.total_delta) < 1e-6

    contrib_map = {c.category: (c.delta, c.contribution_pct) for c in res.contributions}
    assert contrib_map["North"][0] == 1000.0
    assert abs(contrib_map["North"][1] - 50.0) < 1e-6
    assert contrib_map["South"][0] == -1000.0
    assert abs(contrib_map["South"][1] - (-50.0)) < 1e-6


def test_scenario_4_negative_from_zero():
    """TEST 4 — Negative from zero: Prior = 0, Current = -500. Must NOT display +inf%."""
    df_prior = pd.DataFrame({"Profit": [0.0]})
    df_current = pd.DataFrame({"Profit": [-500.0]})

    res = compute_kpi(df_current, df_prior, "Profit")
    assert res.pct_change == float("-inf")
    assert res.pct_change != float("inf")

    formatted = format_pct_change(res.pct_change)
    assert formatted == "n/a (no prior period)"
    assert "inf" not in formatted.lower()


def test_scenario_5_missing_dimension():
    """TEST 5 — Missing dimension: Region North, South, NULL. NULL row must not silently disappear."""
    df_prior = pd.DataFrame({"Region": ["North", "South", None], "Revenue": [100.0, 200.0, 50.0]})
    df_current = pd.DataFrame({"Region": ["North", "South", None], "Revenue": [150.0, 250.0, 80.0]})

    res = compute_dimension_drivers(df_current, df_prior, "Region", "Revenue")
    assert res is not None
    categories = {c.category for c in res.contributions}
    assert "(missing)" in categories

    missing_contrib = next(c for c in res.contributions if c.category == "(missing)")
    assert missing_contrib.delta == 30.0


def test_scenario_6_compact_dates():
    """TEST 6 — Compact dates: 20240101, 20240102. Must be recognized as dates."""
    df = pd.DataFrame({"date_col": ["20240101", "20240102", "20240103"], "sales": [10, 20, 30]})
    clean_df, report = clean_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(clean_df["date_col"])

    classifications = classify_columns(clean_df)
    date_c = next(c for c in classifications if c.name == "date_col")
    assert date_c.role == ROLE_DATE


def test_scenario_7_percent_strings():
    """TEST 7 — Percent strings: '45%', '50%', '12.5%'. Must retain percentage semantics."""
    df = pd.DataFrame({"margin": ["45%", "50%", "12.5%"]})
    clean_df, report = clean_dataframe(df)
    assert pd.api.types.is_numeric_dtype(clean_df["margin"])
    assert abs(clean_df["margin"].iloc[0] - 0.45) < 1e-6
    assert abs(clean_df["margin"].iloc[1] - 0.50) < 1e-6
    assert abs(clean_df["margin"].iloc[2] - 0.125) < 1e-6

    classifications = classify_columns(clean_df)
    margin_c = next(c for c in classifications if c.name == "margin")
    assert margin_c.is_rate is True


def test_scenario_8_dirty_numeric():
    """TEST 8 — Dirty numeric: 100, 200, 'invalid', 300. If coerced, report the failed value count."""
    df = pd.DataFrame({
        "revenue": ["100", "200", "300", "400", "500", "600", "700", "800", "900", "invalid"]
    })
    clean_df, report = clean_dataframe(df)
    assert pd.api.types.is_numeric_dtype(clean_df["revenue"])
    assert clean_df["revenue"].isna().sum() == 1
    assert any("1 value(s) could not be parsed and are now empty" in m for m in report.messages)
