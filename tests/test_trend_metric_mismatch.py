import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from analytics.kpis import select_primary_and_secondary_metrics
from analytics.trend import build_trend_series
from ui.components import build_trend_figure


def test_trend_metric_uses_primary_metric_not_first_column():
    """When metric_cols = ['units', 'revenue', 'cost'] and revenue has the largest total,
    the trend calculation and chart title must strictly reflect revenue, not units."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "units": [10.0, 15.0, 12.0, 14.0, 20.0],          # sum = 71
        "revenue": [1000.0, 1500.0, 1200.0, 1400.0, 2000.0], # sum = 7100 (largest)
        "cost": [500.0, 700.0, 600.0, 800.0, 900.0],      # sum = 3500
    })
    metric_cols = ["units", "revenue", "cost"]

    # 1. Primary metric selection should select revenue
    primary_metric, secondary = select_primary_and_secondary_metrics(df, metric_cols)
    assert primary_metric == "revenue"

    # 2. Trend series built with primary_metric must contain revenue numbers (1000..2000)
    trend = build_trend_series(df, "date", primary_metric)
    assert trend is not None
    assert trend.series["value"].max() == 2000.0
    assert trend.series["value"].min() == 1000.0

    # 3. Chart figure title must be 'Revenue', not 'Units'
    fig = build_trend_figure(trend.series, title=primary_metric)
    assert fig.layout.title.text == "Revenue"
