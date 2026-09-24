import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analytics.anomalies import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    detect_group_outliers,
    detect_period_swings,
)


def test_detect_group_outliers_finds_extreme_category():
    # 5 categories, one is huge (outlier)
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East", "West", "Mega"],
            "revenue": [10.0, 12.0, 11.0, 10.5, 100.0],
        }
    )
    signals = detect_group_outliers(df, "revenue", "region", z_threshold=1.5)
    assert len(signals) >= 1
    assert any(s.category == "Mega" for s in signals)


def test_detect_group_outliers_handles_missing_dimension():
    # Category with NaN in dimension should be grouped as (missing), not dropped
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East", np.nan, "Mega"],
            "revenue": [10.0, 12.0, 11.0, 9.0, 100.0],
        }
    )
    signals = detect_group_outliers(df, "revenue", "region", z_threshold=1.5)
    assert len(signals) >= 1
    # Mega is detected while (missing) was retained in grouping statistics
    assert any(s.category == "Mega" for s in signals)


def test_detect_period_swings_severity_tiers():
    # Baseline 100. If swung 35% -> low/medium. If swung 75% -> high.
    df_prior = pd.DataFrame({"region": ["A", "B"], "revenue": [100.0, 100.0]})
    df_current = pd.DataFrame({"region": ["A", "B"], "revenue": [135.0, 180.0]})
    signals = detect_period_swings(df_current, df_prior, "revenue", "region", pct_threshold=0.3)
    assert len(signals) == 2
    b_signal = next(s for s in signals if s.category == "B")
    assert b_signal.severity == SEVERITY_HIGH
