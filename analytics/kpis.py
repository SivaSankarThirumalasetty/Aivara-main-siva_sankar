"""
analytics/kpis.py — summary KPI computation.

Auto-selects a primary metric plus up to 3 secondary metrics, and computes
current/prior period totals + deltas for each. Also derives a margin-like
ratio KPI when a cost/revenue-shaped pair of metrics is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

COST_NAME_PATTERN = ("cost", "expense", "spend")
REVENUE_NAME_PATTERN = ("revenue", "sales", "income", "gmv")
RATE_NAME_PATTERN = re.compile(r"(rate|pct|percent|ratio|avg|average|score|margin)", re.I)


@dataclass
class KPIResult:
    name: str
    unit: str | None
    current_value: float
    prior_value: float | None
    pct_change: float | None
    abs_change: float | None
    is_derived_ratio: bool = False


def is_rate_metric(metric_name: str) -> bool:
    return bool(RATE_NAME_PATTERN.search(metric_name))


def select_primary_and_secondary_metrics(
    df: pd.DataFrame, metric_columns: list[str], max_secondary: int = 3
) -> tuple[str | None, list[str]]:
    """Primary = metric column with the largest sum of absolute values.
    Secondary = next `max_secondary` metrics by the same measure."""
    if not metric_columns:
        return None, []

    sums = {
        col: df[col].abs().sum(skipna=True)
        for col in metric_columns
        if pd.api.types.is_numeric_dtype(df[col])
    }
    ranked = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None, []

    primary = ranked[0][0]
    secondary = [name for name, _ in ranked[1 : 1 + max_secondary]]
    return primary, secondary


def _totals(
    series_current: pd.Series, series_prior: pd.Series | None, is_rate: bool = False
) -> tuple[float, float | None]:
    if is_rate:
        current_total = float(series_current.mean(skipna=True))
        prior_total = float(series_prior.mean(skipna=True)) if series_prior is not None else None
    else:
        current_total = float(series_current.sum(skipna=True))
        prior_total = float(series_prior.sum(skipna=True)) if series_prior is not None else None
    return current_total, prior_total


def compute_kpi(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame | None,
    metric_col: str,
) -> KPIResult:
    is_rate = is_rate_metric(metric_col)
    current_total, prior_total = _totals(
        df_current[metric_col],
        df_prior[metric_col] if df_prior is not None else None,
        is_rate=is_rate,
    )

    pct_change = None
    abs_change = None
    if prior_total is not None:
        abs_change = current_total - prior_total
        if prior_total != 0:
            pct_change = (abs_change / abs(prior_total)) * 100
        elif current_total != 0:
            pct_change = float("inf") if current_total > 0 else float("-inf")
        else:
            pct_change = 0.0

    return KPIResult(
        name=metric_col,
        unit=None,
        current_value=current_total,
        prior_value=prior_total,
        pct_change=pct_change,
        abs_change=abs_change,
    )


def find_margin_pair(metric_columns: list[str]) -> tuple[str, str] | None:
    """Look for a (cost, revenue) shaped pair of metric names, e.g.
    ('cost', 'revenue') or ('total_cost', 'total_sales')."""
    cost_col = next(
        (c for c in metric_columns if any(p in c.lower() for p in COST_NAME_PATTERN)), None
    )
    revenue_col = next(
        (c for c in metric_columns if any(p in c.lower() for p in REVENUE_NAME_PATTERN)), None
    )
    if cost_col and revenue_col and cost_col != revenue_col:
        return cost_col, revenue_col
    return None


def compute_margin_kpi(
    df_current: pd.DataFrame, df_prior: pd.DataFrame | None, cost_col: str, revenue_col: str
) -> KPIResult | None:
    cur_rev = float(df_current[revenue_col].sum(skipna=True))
    cur_cost = float(df_current[cost_col].sum(skipna=True))
    if cur_rev == 0:
        return None
    cur_margin = (cur_rev - cur_cost) / cur_rev * 100

    prior_margin = None
    pct_change = None
    abs_change = None
    if df_prior is not None and len(df_prior):
        prior_rev = float(df_prior[revenue_col].sum(skipna=True))
        prior_cost = float(df_prior[cost_col].sum(skipna=True))
        if prior_rev != 0:
            prior_margin = (prior_rev - prior_cost) / prior_rev * 100
            abs_change = cur_margin - prior_margin
            pct_change = abs_change  # margin deltas are typically reported in points, not %-of-%

    return KPIResult(
        name=f"{revenue_col}_margin",
        unit="%",
        current_value=cur_margin,
        prior_value=prior_margin,
        pct_change=pct_change,
        abs_change=abs_change,
        is_derived_ratio=True,
    )


def compute_all_kpis(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame | None,
    metric_columns: list[str],
    max_secondary: int = 3,
) -> dict[str, KPIResult]:
    primary, secondary = select_primary_and_secondary_metrics(
        df_current, metric_columns, max_secondary
    )
    results: dict[str, KPIResult] = {}
    if primary:
        results[primary] = compute_kpi(df_current, df_prior, primary)
    for col in secondary:
        results[col] = compute_kpi(df_current, df_prior, col)

    margin_pair = find_margin_pair(metric_columns)
    if margin_pair:
        margin_kpi = compute_margin_kpi(df_current, df_prior, *margin_pair)
        if margin_kpi:
            results[margin_kpi.name] = margin_kpi

    return results
