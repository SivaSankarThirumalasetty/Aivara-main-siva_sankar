import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analytics.drivers import compute_dimension_drivers, rank_dimensions_by_explanatory_power


def test_dimension_drivers_hand_computed():
    # Prior period totals: North=100, South=100 (total 200)
    # Current period totals: North=150, South=110 (total 260)
    # Total delta = 60. North contributes +50 (83.3%), South contributes +10 (16.7%).
    df_prior = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 100]})
    df_current = pd.DataFrame({"region": ["North", "South"], "revenue": [150, 110]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    assert abs(result.total_delta - 60) < 1e-6

    contrib_by_cat = {c.category: c.contribution_pct for c in result.contributions}
    assert abs(contrib_by_cat["North"] - (50 / 60 * 100)) < 1e-6
    assert abs(contrib_by_cat["South"] - (10 / 60 * 100)) < 1e-6
    assert result.top_positive[0].category == "North"


def test_dimension_drivers_net_delta_zero():
    """When net change across categories is exactly zero (North +1000, South -1000),
    neither category must be reported as 0.0%. Both must reflect their share of gross movement."""
    df_prior = pd.DataFrame({"region": ["North", "South"], "revenue": [1000.0, 2000.0]})
    df_current = pd.DataFrame({"region": ["North", "South"], "revenue": [2000.0, 1000.0]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    assert abs(result.total_delta) < 1e-6

    contrib_by_cat = {c.category: c.contribution_pct for c in result.contributions}
    # Sum of abs deltas = 2000. North is +1000 (+50.0%), South is -1000 (-50.0%)
    assert abs(contrib_by_cat["North"] - 50.0) < 1e-6
    assert abs(contrib_by_cat["South"] - (-50.0)) < 1e-6
    assert result.concentration_score > 0.0


def test_dimension_drivers_near_zero_net_delta_safe():
    """When total net delta is tiny (e.g. 0.01) but individual deltas are large (+1000.01, -1000.00),
    contribution percentages must NOT blow up into absurd numbers like +5,000,000%."""
    df_prior = pd.DataFrame({"region": ["North", "South"], "revenue": [1000.0, 2000.0]})
    df_current = pd.DataFrame({"region": ["North", "South"], "revenue": [2000.01, 1000.0]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    contrib_by_cat = {c.category: c.contribution_pct for c in result.contributions}
    # Both contributions must be bounded in [-100%, +100%]
    assert 40.0 < contrib_by_cat["North"] <= 60.0
    assert -60.0 <= contrib_by_cat["South"] < -40.0


def test_dimension_drivers_all_positive_deltas():
    df_prior = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100.0, 100.0, 100.0]})
    df_current = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [120.0, 130.0, 150.0]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    assert result.total_delta == 100.0
    contrib_by_cat = {c.category: c.contribution_pct for c in result.contributions}
    assert abs(contrib_by_cat["A"] - 20.0) < 1e-6
    assert abs(contrib_by_cat["B"] - 30.0) < 1e-6
    assert abs(contrib_by_cat["C"] - 50.0) < 1e-6


def test_dimension_drivers_all_negative_deltas():
    df_prior = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [150.0, 130.0, 120.0]})
    df_current = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100.0, 100.0, 100.0]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    assert result.total_delta == -100.0
    contrib_by_cat = {c.category: c.contribution_pct for c in result.contributions}
    assert abs(contrib_by_cat["A"] - (-50.0)) < 1e-6
    assert abs(contrib_by_cat["B"] - (-30.0)) < 1e-6
    assert abs(contrib_by_cat["C"] - (-20.0)) < 1e-6


def test_dimension_drivers_missing_dimension_bucket():
    """Missing / NaN dimension values must be assigned to '(missing)' rather than silently dropped."""
    df_prior = pd.DataFrame({"region": ["North", None], "revenue": [100.0, 50.0]})
    df_current = pd.DataFrame({"region": ["North", None], "revenue": [150.0, 80.0]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    cats = {c.category for c in result.contributions}
    assert "(missing)" in cats
    contrib_by_cat = {c.category: c.contribution_pct for c in result.contributions}
    # Total delta = 50 + 30 = 80. (missing) delta = +30 (37.5%)
    assert abs(contrib_by_cat["(missing)"] - 37.5) < 1e-6


def test_dimension_drivers_handles_new_and_dropped_categories():
    df_prior = pd.DataFrame({"region": ["North"], "revenue": [100]})
    df_current = pd.DataFrame({"region": ["North", "West"], "revenue": [100, 50]})

    result = compute_dimension_drivers(df_current, df_prior, "region", "revenue")
    assert result is not None
    cats = {c.category for c in result.contributions}
    assert cats == {"North", "West"}


def test_no_prior_period_returns_none():
    df_current = pd.DataFrame({"region": ["North"], "revenue": [100]})
    result = compute_dimension_drivers(df_current, None, "region", "revenue")
    assert result is None


def test_rank_dimensions_by_concentration():
    df_prior = pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "product": ["A", "B", "C", "D"],
            "revenue": [100, 100, 100, 100],
        }
    )
    df_current = pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "product": ["A", "B", "C", "D"],
            "revenue": [140, 100, 100, 100],
        }
    )
    ranked = rank_dimensions_by_explanatory_power(df_current, df_prior, ["region", "product"], "revenue")
    assert ranked[0].dimension in ("region", "product")
    assert ranked[0].concentration_score >= ranked[-1].concentration_score
