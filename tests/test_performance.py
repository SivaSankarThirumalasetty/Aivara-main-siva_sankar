import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.anomalies import compute_risk_signals
from analytics.drivers import rank_dimensions_by_explanatory_power
from analytics.kpis import compute_all_kpis
from analytics.trend import build_trend_series, split_current_prior
from ingestion.classifier import classify_columns
from ingestion.cleaner import clean_dataframe


def _generate_synthetic_df(n_rows: int) -> pd.DataFrame:
    np.random.seed(42)
    n_periods = max(5, min(100, n_rows // 10))
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")
    regions = ["North", "South", "East", "West", "Central"]
    products = [f"Product_{i}" for i in range(20)]

    return pd.DataFrame({
        "date": np.random.choice(dates, n_rows),
        "region": np.random.choice(regions, n_rows),
        "product": np.random.choice(products, n_rows),
        "revenue": np.random.exponential(100.0, n_rows),
        "cost": np.random.exponential(60.0, n_rows),
        "units": np.random.randint(1, 50, n_rows),
        "conversion_rate": np.random.uniform(0.01, 0.25, n_rows),
    })


def test_performance_small_100_rows():
    df = _generate_synthetic_df(100)
    t0 = time.perf_counter()

    clean_df, report = clean_dataframe(df)
    classifications = classify_columns(clean_df)
    trend = build_trend_series(clean_df, "date", "revenue")
    df_cur, df_pri = split_current_prior(clean_df, "date", trend)
    kpis = compute_all_kpis(df_cur, df_pri, ["revenue", "cost", "units", "conversion_rate"])
    drivers = rank_dimensions_by_explanatory_power(df_cur, df_pri, ["region", "product"], "revenue")
    risks = compute_risk_signals(df_cur, df_pri, ["revenue"], ["region"])

    duration = time.perf_counter() - t0
    assert duration < 1.0  # sub-second for 100 rows
    assert "revenue" in kpis
    assert len(drivers) > 0


def test_performance_medium_10000_rows():
    df = _generate_synthetic_df(10_000)
    t0 = time.perf_counter()

    clean_df, report = clean_dataframe(df)
    classifications = classify_columns(clean_df)
    trend = build_trend_series(clean_df, "date", "revenue")
    df_cur, df_pri = split_current_prior(clean_df, "date", trend)
    kpis = compute_all_kpis(df_cur, df_pri, ["revenue", "cost", "units", "conversion_rate"])
    drivers = rank_dimensions_by_explanatory_power(df_cur, df_pri, ["region", "product"], "revenue")
    risks = compute_risk_signals(df_cur, df_pri, ["revenue"], ["region"])

    duration = time.perf_counter() - t0
    assert duration < 3.0  # under 3 seconds for 10,000 rows
    assert "revenue" in kpis


def test_performance_large_100000_rows():
    df = _generate_synthetic_df(100_000)
    t0 = time.perf_counter()

    clean_df, report = clean_dataframe(df)
    classifications = classify_columns(clean_df)
    trend = build_trend_series(clean_df, "date", "revenue")
    df_cur, df_pri = split_current_prior(clean_df, "date", trend)
    kpis = compute_all_kpis(df_cur, df_pri, ["revenue", "cost", "units", "conversion_rate"])
    drivers = rank_dimensions_by_explanatory_power(df_cur, df_pri, ["region", "product"], "revenue")

    duration = time.perf_counter() - t0
    assert duration < 8.0  # under 8 seconds for 100,000 rows
    assert "revenue" in kpis
