"""
analytics/drivers.py — dimension decomposition ("what explains the change").

For the primary KPI and each classified dimension column, computes each
category's contribution to the period-over-period change, ranks dimensions
by how well they "explain" the change (concentration of contribution), and
surfaces top positive/negative contributors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

RATE_NAME_PATTERN = re.compile(r"(rate|pct|percent|ratio|avg|average|score|margin)", re.I)


@dataclass
class CategoryContribution:
    category: str
    current_value: float
    prior_value: float
    delta: float
    contribution_pct: float  # share of total delta, can be negative


@dataclass
class DimensionDriverResult:
    dimension: str
    total_delta: float
    contributions: list[CategoryContribution] = field(default_factory=list)
    concentration_score: float = 0.0  # higher = a few categories explain most of the change

    @property
    def top_positive(self) -> list[CategoryContribution]:
        pos = [c for c in self.contributions if c.delta > 0]
        return sorted(pos, key=lambda c: c.delta, reverse=True)[:3]

    @property
    def top_negative(self) -> list[CategoryContribution]:
        neg = [c for c in self.contributions if c.delta < 0]
        return sorted(neg, key=lambda c: c.delta)[:2]


def _concentration_score(contributions: list[CategoryContribution]) -> float:
    """Herfindahl-style concentration of |contribution_pct| across
    categories: higher means fewer categories explain most of the change."""
    total_abs = sum(abs(c.contribution_pct) for c in contributions)
    if total_abs == 0 or not contributions:
        return 0.0
    shares = [abs(c.contribution_pct) / total_abs for c in contributions]
    return sum(s**2 for s in shares)  # in [1/n, 1]


def compute_dimension_drivers(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame | None,
    dimension_col: str,
    metric_col: str,
) -> DimensionDriverResult | None:
    if df_prior is None or df_prior.empty or df_current.empty:
        return None

    cur_dim = df_current[dimension_col].fillna("(missing)").astype(str)
    prior_dim = df_prior[dimension_col].fillna("(missing)").astype(str)

    is_rate = bool(RATE_NAME_PATTERN.search(metric_col))
    if is_rate:
        cur_grouped = df_current.groupby(cur_dim)[metric_col].mean(numeric_only=True)
        prior_grouped = df_prior.groupby(prior_dim)[metric_col].mean(numeric_only=True)
        total_delta = float(df_current[metric_col].mean(skipna=True) - df_prior[metric_col].mean(skipna=True))
    else:
        cur_grouped = df_current.groupby(cur_dim)[metric_col].sum(numeric_only=True)
        prior_grouped = df_prior.groupby(prior_dim)[metric_col].sum(numeric_only=True)
        total_delta = float(cur_grouped.sum() - prior_grouped.sum())

    all_categories = sorted(set(cur_grouped.index) | set(prior_grouped.index))
    deltas: dict[str, tuple[float, float, float]] = {}
    for cat in all_categories:
        cur_val = float(cur_grouped.get(cat, 0.0))
        prior_val = float(prior_grouped.get(cat, 0.0))
        deltas[cat] = (cur_val, prior_val, cur_val - prior_val)

    sum_abs_delta = sum(abs(d[2]) for d in deltas.values())
    denominator = sum_abs_delta if sum_abs_delta > 1e-9 else 0.0

    contributions = []
    for cat in all_categories:
        cur_val, prior_val, delta = deltas[cat]
        contribution_pct = (delta / denominator * 100.0) if denominator != 0.0 else 0.0
        contributions.append(
            CategoryContribution(
                category=str(cat),
                current_value=cur_val,
                prior_value=prior_val,
                delta=delta,
                contribution_pct=contribution_pct,
            )
        )

    result = DimensionDriverResult(dimension=dimension_col, total_delta=total_delta, contributions=contributions)
    result.concentration_score = _concentration_score(contributions)
    return result


def rank_dimensions_by_explanatory_power(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame | None,
    dimension_columns: list[str],
    metric_col: str,
) -> list[DimensionDriverResult]:
    results = []
    for dim in dimension_columns:
        r = compute_dimension_drivers(df_current, df_prior, dim, metric_col)
        if r is not None and r.contributions:
            results.append(r)
    return sorted(results, key=lambda r: r.concentration_score, reverse=True)
