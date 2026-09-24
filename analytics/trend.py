"""
analytics/trend.py — time series aggregation and period-over-period split.

Requires a classified `date` column. If none exists, the caller should skip
trend/period-comparison and fall back to a single-snapshot view rather than
erroring (Section 8.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

GRANULARITY_ORDER = ["D", "W", "M"]
RATE_NAME_PATTERN = re.compile(r"(rate|pct|percent|ratio|avg|average|score|margin)", re.I)


@dataclass
class TrendResult:
    granularity: str
    series: pd.DataFrame  # columns: [period, value]
    current_period_start: pd.Timestamp
    prior_period_start: pd.Timestamp | None


def _ensure_datetime(dates: pd.Series) -> pd.Series:
    """Coerce a series to datetime64, handling numeric (Excel serial /
    YYYYMMDD integer) values, returning a Series with NaT for unparseable
    rows. Never raises."""
    if pd.api.types.is_datetime64_any_dtype(dates):
        return dates
    # Try generic parsing first (handles ISO strings, YYYYMMDD strings, etc.)
    try:
        converted = pd.to_datetime(dates, errors="coerce")
        # If too many NaTs, try treating numeric values as Excel serial dates
        nat_frac = converted.isna().mean()
        if nat_frac > 0.5 and pd.api.types.is_numeric_dtype(dates):
            # Excel serial: days since 1899-12-30
            try:
                excel_converted = pd.to_datetime(
                    dates, unit="D", origin="1899-12-30", errors="coerce"
                )
                if excel_converted.notna().mean() > converted.notna().mean():
                    converted = excel_converted
            except Exception:
                pass
        return converted
    except Exception:
        return pd.Series([pd.NaT] * len(dates), index=dates.index)


def pick_granularity(dates: pd.Series) -> str:
    """Pick day/week/month based on both the overall span AND row density —
    a dataset spanning 3 weeks with one data point per week should bucket by
    week, not by day, even though the total span is under 31 days."""
    dates = _ensure_datetime(dates).dropna()
    if dates.empty:
        return "D"

    distinct_sorted = pd.Series(dates.unique()).sort_values()
    # After unique(), values may be numpy datetime64 — convert to Timestamps
    # so .days attribute works reliably.
    distinct_sorted = pd.to_datetime(distinct_sorted)

    if len(distinct_sorted) < 2:
        return "D"

    span_days = (distinct_sorted.iloc[-1] - distinct_sorted.iloc[0]).days or 1
    gaps = distinct_sorted.diff().dropna().dt.days
    median_gap = gaps.median() if not gaps.empty else 1

    if median_gap <= 2 and span_days <= 31:
        return "D"
    if median_gap <= 10 or span_days <= 180:
        return "W"
    return "M"


def build_trend_series(
    df: pd.DataFrame, date_col: str, metric_col: str, granularity: str | None = None
) -> TrendResult | None:
    working = df[[date_col, metric_col]].copy()
    # Always coerce the date column to proper datetime so resample() works.
    working[date_col] = _ensure_datetime(working[date_col])
    working = working.dropna(subset=[date_col])
    if working.empty:
        return None

    if granularity is None:
        granularity = pick_granularity(working[date_col])

    freq_map = {"D": "D", "W": "W-MON", "M": "MS"}
    working = working.set_index(date_col)
    is_rate = bool(RATE_NAME_PATTERN.search(metric_col))
    resampler = working[metric_col].resample(freq_map[granularity])
    resampled = (resampler.mean() if is_rate else resampler.sum()).reset_index()
    resampled.columns = ["period", "value"]
    resampled = resampled.sort_values("period").reset_index(drop=True)

    if resampled.empty:
        return None

    current_start = resampled["period"].iloc[-1]
    prior_start = resampled["period"].iloc[-2] if len(resampled) >= 2 else None

    return TrendResult(
        granularity=granularity,
        series=resampled,
        current_period_start=current_start,
        prior_period_start=prior_start,
    )


def split_current_prior(
    df: pd.DataFrame, date_col: str, trend: TrendResult
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split the raw (row-level) dataframe into current-period and
    prior-period subsets, based on the bucket boundaries used for the trend
    series. This is what feeds KPI and driver computation so all three stay
    consistent about what \"current period\" means."""
    granularity = trend.granularity
    current_start = trend.current_period_start
    prior_start = trend.prior_period_start

    offset = {
        "D": pd.DateOffset(days=1),
        "W": pd.DateOffset(weeks=1),
        "M": pd.DateOffset(months=1),
    }.get(granularity, pd.DateOffset(days=1))

    # Ensure the df date column is datetime for comparison
    date_series = _ensure_datetime(df[date_col])

    current_end = current_start + offset
    current_mask = (date_series >= current_start) & (date_series < current_end)
    df_current = df.loc[current_mask]

    df_prior = None
    if prior_start is not None:
        prior_end = prior_start + offset
        prior_mask = (date_series >= prior_start) & (date_series < prior_end)
        df_prior = df.loc[prior_mask]

    return df_current, df_prior


def period_label(granularity: str) -> str:
    return {
        "D": "today vs yesterday",
        "W": "this week vs last week",
        "M": "this month vs last month",
    }.get(granularity, "this period vs last period")
