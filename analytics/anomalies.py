"""
analytics/anomalies.py — simple, explainable outlier / risk-signal detection.

No black-box ML for v1: z-score / IQR outlier detection per metric within a
dimension group, plus sudden period-over-period swings beyond a threshold.
Each flagged item becomes one "risk signal".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ANOMALY_PCT_SWING_THRESHOLD, ANOMALY_ZSCORE_THRESHOLD

SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"


@dataclass
class RiskSignal:
    description: str
    severity: str
    metric: str
    dimension: str | None = None
    category: str | None = None
    value: float | None = None


def detect_group_outliers(
    df: pd.DataFrame, metric_col: str, dimension_col: str, z_threshold: float = ANOMALY_ZSCORE_THRESHOLD
) -> list[RiskSignal]:
    """Flag categories within `dimension_col` whose `metric_col` total is a
    z-score outlier relative to the other categories."""
    signals: list[RiskSignal] = []
    dim_series = df[dimension_col].fillna("(missing)").astype(str)
    grouped = df.groupby(dim_series)[metric_col].sum(numeric_only=True)
    if len(grouped) < 3:
        return signals

    mean = grouped.mean()
    std = grouped.std(ddof=0)
    if std == 0 or pd.isna(std):
        return signals

    z_scores = (grouped - mean) / std
    for cat, z in z_scores.items():
        if abs(z) >= z_threshold:
            if abs(z) >= z_threshold * 2.0:
                severity = SEVERITY_HIGH
            elif abs(z) >= z_threshold * 1.5:
                severity = SEVERITY_MEDIUM
            else:
                severity = SEVERITY_LOW
            direction = "unusually high" if z > 0 else "unusually low"
            signals.append(
                RiskSignal(
                    description=f"'{cat}' has {direction} {metric_col} relative to other {dimension_col} groups (z={z:.1f}).",
                    severity=severity,
                    metric=metric_col,
                    dimension=dimension_col,
                    category=str(cat),
                    value=float(grouped[cat]),
                )
            )
    return signals


def detect_period_swings(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame | None,
    metric_col: str,
    dimension_col: str | None = None,
    pct_threshold: float = ANOMALY_PCT_SWING_THRESHOLD,
) -> list[RiskSignal]:
    """Flag sudden period-over-period swings beyond a flat % threshold
    (used as the short-history fallback when we don't have enough periods
    to compute a proper historical volatility band)."""
    signals: list[RiskSignal] = []
    if df_prior is None or df_prior.empty or df_current.empty:
        return signals

    if dimension_col is None:
        cur_total = float(df_current[metric_col].sum(skipna=True))
        prior_total = float(df_prior[metric_col].sum(skipna=True))
        if prior_total == 0:
            return signals
        pct_change = (cur_total - prior_total) / abs(prior_total)
        if abs(pct_change) >= pct_threshold:
            if abs(pct_change) >= pct_threshold * 2.0:
                severity = SEVERITY_HIGH
            elif abs(pct_change) >= pct_threshold * 1.5:
                severity = SEVERITY_MEDIUM
            else:
                severity = SEVERITY_LOW
            signals.append(
                RiskSignal(
                    description=f"{metric_col} swung {pct_change:+.0%} period-over-period.",
                    severity=severity,
                    metric=metric_col,
                    value=cur_total,
                )
            )
        return signals

    cur_dim = df_current[dimension_col].fillna("(missing)").astype(str)
    prior_dim = df_prior[dimension_col].fillna("(missing)").astype(str)
    cur_grouped = df_current.groupby(cur_dim)[metric_col].sum(numeric_only=True)
    prior_grouped = df_prior.groupby(prior_dim)[metric_col].sum(numeric_only=True)
    common = set(cur_grouped.index) & set(prior_grouped.index)
    for cat in common:
        prior_val = float(prior_grouped[cat])
        cur_val = float(cur_grouped[cat])
        if prior_val == 0:
            continue
        pct_change = (cur_val - prior_val) / abs(prior_val)
        if abs(pct_change) >= pct_threshold:
            if abs(pct_change) >= pct_threshold * 2.0:
                severity = SEVERITY_HIGH
            elif abs(pct_change) >= pct_threshold * 1.5:
                severity = SEVERITY_MEDIUM
            else:
                severity = SEVERITY_LOW
            signals.append(
                RiskSignal(
                    description=f"'{cat}' {metric_col} swung {pct_change:+.0%} period-over-period.",
                    severity=severity,
                    metric=metric_col,
                    dimension=dimension_col,
                    category=str(cat),
                    value=cur_val,
                )
            )
    return signals


def compute_risk_signals(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame | None,
    metric_columns: list[str],
    dimension_columns: list[str],
) -> list[RiskSignal]:
    all_signals: list[RiskSignal] = []
    for metric in metric_columns:
        all_signals.extend(detect_period_swings(df_current, df_prior, metric))
        for dim in dimension_columns:
            all_signals.extend(detect_group_outliers(df_current, metric, dim))
            all_signals.extend(detect_period_swings(df_current, df_prior, metric, dim))

    # De-duplicate near-identical descriptions.
    seen = set()
    deduped = []
    for s in all_signals:
        key = (s.metric, s.dimension, s.category)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped
