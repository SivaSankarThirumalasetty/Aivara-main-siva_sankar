import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pandas as pd

from analytics.kpis import (
    compute_all_kpis,
    compute_kpi,
    compute_margin_kpi,
    find_margin_pair,
    is_rate_metric,
    select_primary_and_secondary_metrics,
)
from ui.components import format_pct_change


def test_select_primary_metric_by_largest_absolute_sum():
    df = pd.DataFrame({"revenue": [100, 200, 300], "orders": [1, 2, 3], "cost": [50, 60, 70]})
    primary, secondary = select_primary_and_secondary_metrics(df, ["revenue", "orders", "cost"])
    assert primary == "revenue"
    assert set(secondary) == {"orders", "cost"}


def test_compute_kpi_pct_change():
    df_current = pd.DataFrame({"revenue": [100, 200]})  # total 300
    df_prior = pd.DataFrame({"revenue": [100, 100]})  # total 200
    result = compute_kpi(df_current, df_prior, "revenue")
    assert result.current_value == 300
    assert result.prior_value == 200
    assert abs(result.pct_change - 50.0) < 1e-6


def test_compute_kpi_no_prior_period():
    df_current = pd.DataFrame({"revenue": [100, 200]})
    result = compute_kpi(df_current, None, "revenue")
    assert result.prior_value is None
    assert result.pct_change is None


def test_positive_swing_from_zero_pct_change():
    df_current = pd.DataFrame({"revenue": [500]})
    df_prior = pd.DataFrame({"revenue": [0]})
    result = compute_kpi(df_current, df_prior, "revenue")
    assert result.pct_change == float("inf")


def test_negative_swing_from_zero_pct_change():
    df_current = pd.DataFrame({"profit": [-500]})
    df_prior = pd.DataFrame({"profit": [0]})
    result = compute_kpi(df_current, df_prior, "profit")
    assert result.pct_change == float("-inf")


def test_zero_to_zero_pct_change():
    df_current = pd.DataFrame({"revenue": [0]})
    df_prior = pd.DataFrame({"revenue": [0]})
    result = compute_kpi(df_current, df_prior, "revenue")
    assert result.pct_change == 0.0


def test_format_pct_change_never_outputs_infinity_strings():
    """Verify format_pct_change produces human-readable text and never 'inf%' or '+inf%'."""
    assert format_pct_change(float("inf")) == "new"
    assert format_pct_change(float("-inf")) == "n/a (no prior period)"
    assert format_pct_change(float("inf"), with_arrow=True) == "new"
    assert format_pct_change(float("-inf"), with_arrow=True) == "n/a (no prior period)"
    assert format_pct_change(float("inf"), with_sign=True) == "new"
    assert format_pct_change(float("-inf"), with_sign=True) == "n/a (no prior period)"
    assert format_pct_change(None) is None
    assert format_pct_change(15.5, with_sign=True) == "+15.5%"
    assert format_pct_change(-12.3, with_sign=True) == "-12.3%"
    assert format_pct_change(0.0, with_sign=True) == "+0.0%"


def test_margin_pair_detection_and_computation():
    df_current = pd.DataFrame({"revenue": [1000], "cost": [600]})
    df_prior = pd.DataFrame({"revenue": [800], "cost": [560]})
    pair = find_margin_pair(["revenue", "cost", "orders"])
    assert pair == ("cost", "revenue")
    margin = compute_margin_kpi(df_current, df_prior, "cost", "revenue")
    assert margin is not None
    assert abs(margin.current_value - 40.0) < 1e-6  # (1000-600)/1000
    assert abs(margin.prior_value - 30.0) < 1e-6  # (800-560)/800


def test_compute_all_kpis_includes_margin_when_present():
    df_current = pd.DataFrame({"revenue": [1000], "cost": [600], "orders": [10]})
    df_prior = pd.DataFrame({"revenue": [800], "cost": [560], "orders": [8]})
    results = compute_all_kpis(df_current, df_prior, ["revenue", "cost", "orders"])
    assert "revenue" in results
    assert any(k.endswith("_margin") for k in results)


def test_rate_metric_aggregated_by_mean_not_sum():
    # Example 1 from brief: daily conversion rates = 10%, 20%, 30% -> must report 20%, NOT 60%
    df_current = pd.DataFrame({"conversion_rate": [10.0, 20.0, 30.0]})
    df_prior = pd.DataFrame({"conversion_rate": [15.0, 15.0, 15.0]})
    result = compute_kpi(df_current, df_prior, "conversion_rate")
    assert abs(result.current_value - 20.0) < 1e-6
    assert result.current_value != 60.0
    assert abs(result.prior_value - 15.0) < 1e-6


def test_average_order_value_aggregated_by_mean_not_sum():
    # Example 2 from brief: AOV must not be treated as additive revenue
    df_current = pd.DataFrame({"avg_order_value": [100.0, 150.0, 200.0]})
    df_prior = pd.DataFrame({"avg_order_value": [100.0, 100.0, 100.0]})
    result = compute_kpi(df_current, df_prior, "avg_order_value")
    assert abs(result.current_value - 150.0) < 1e-6
    assert result.current_value != 450.0
