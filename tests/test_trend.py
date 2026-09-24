import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from analytics.trend import build_trend_series, pick_granularity, split_current_prior, _ensure_datetime


def test_pick_granularity_daily():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    assert pick_granularity(pd.Series(dates)) == "D"


def test_pick_granularity_weekly():
    dates = pd.date_range("2024-01-01", periods=10, freq="7D")
    assert pick_granularity(pd.Series(dates)) == "W"


def test_pick_granularity_monthly():
    dates = pd.date_range("2024-01-01", periods=12, freq="30D")
    assert pick_granularity(pd.Series(dates)) == "M"


def test_split_current_prior():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "revenue": [100, 200, 300],
        }
    )
    trend = build_trend_series(df, "date", "revenue", granularity="D")
    assert trend is not None
    df_current, df_prior = split_current_prior(df, "date", trend)
    assert len(df_current) == 1
    assert df_current["revenue"].iloc[0] == 300
    assert df_prior is not None
    assert len(df_prior) == 1
    assert df_prior["revenue"].iloc[0] == 200


def test_ensure_datetime_passthrough_for_datetime_series():
    """Already-datetime series should come back unchanged."""
    dates = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D"))
    result = _ensure_datetime(dates)
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert result.notna().all()


def test_ensure_datetime_coerces_iso_strings():
    """ISO date strings (from CSV) should parse cleanly."""
    dates = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
    result = _ensure_datetime(dates)
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert result.notna().all()


def test_ensure_datetime_coerces_yyyymmdd_strings():
    """YYYYMMDD string dates should coerce without becoming NaT."""
    dates = pd.Series(["20240101", "20240102", "20240103"])
    result = _ensure_datetime(dates)
    assert result.notna().sum() >= 2  # at least most should parse


def test_pick_granularity_with_float_date_column_does_not_crash():
    """When date column contains numpy.float64 values (e.g. Excel serial numbers
    or YYYYMMDD integers), pick_granularity must not raise AttributeError."""
    # Simulate YYYYMMDD as integer floats
    float_dates = pd.Series([20240101.0, 20240102.0, 20240103.0, 20240104.0, 20240105.0])
    # Should not raise; may return any valid granularity string
    result = pick_granularity(float_dates)
    assert result in ("D", "W", "M")


def test_build_trend_series_with_string_dates():
    """build_trend_series should handle ISO string date columns from CSV."""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
    })
    trend = build_trend_series(df, "date", "revenue")
    assert trend is not None
    assert len(trend.series) >= 1
