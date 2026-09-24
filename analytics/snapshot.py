"""
analytics/snapshot.py — analytics for Cross-Sectional Snapshots and Project Portfolios.

Provides:
1. Snapshot Rankings, Pareto (ABC) concentration, Outlier detection.
2. Project Portfolio Completion, Team Workload, and At-Risk Task flagging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from analytics.kpis import KPIResult, is_rate_metric


@dataclass
class EntityContribution:
    entity: str
    value: float
    pct_share: float


@dataclass
class SnapshotAnalyticsResult:
    total_items: int
    kpis: dict[str, KPIResult]
    top_entities: list[EntityContribution]
    bottom_entities: list[EntityContribution]
    pareto_text: str
    outliers: list[dict]
    group_rollups: dict[str, pd.DataFrame]


@dataclass
class ProjectAnalyticsResult:
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    unstarted_tasks: int
    completion_rate_pct: float
    total_days: float
    kpis: dict[str, KPIResult]
    project_rollups: pd.DataFrame
    assignee_workload: pd.DataFrame
    at_risk_tasks: list[dict]


def compute_snapshot_analytics(
    df: pd.DataFrame,
    entity_col: str | None,
    primary_metric: str,
    secondary_metrics: list[str],
    grouping_cols: list[str] | None = None,
) -> SnapshotAnalyticsResult:
    """Compute rich rankings, Pareto concentration, and outliers for a
    cross-sectional snapshot (e.g. inventory, product catalog, sales leaderboard)."""
    total_items = len(df)
    kpis: dict[str, KPIResult] = {}

    all_metrics = [primary_metric] + [m for m in secondary_metrics if m != primary_metric]
    for m in all_metrics:
        if m not in df.columns or not pd.api.types.is_numeric_dtype(df[m]):
            continue
        is_rate = is_rate_metric(m)
        val = float(df[m].mean(skipna=True)) if is_rate else float(df[m].sum(skipna=True))
        unit = "%" if is_rate else None
        kpis[m] = KPIResult(
            name=m,
            unit=unit,
            current_value=val,
            prior_value=None,
            pct_change=None,
            abs_change=None,
            is_derived_ratio=is_rate,
        )

    # Entity rankings by primary_metric
    top_entities: list[EntityContribution] = []
    bottom_entities: list[EntityContribution] = []
    pareto_text = ""

    entity = entity_col if (entity_col and entity_col in df.columns) else df.columns[0]
    if primary_metric in df.columns and pd.api.types.is_numeric_dtype(df[primary_metric]):
        is_primary_rate = is_rate_metric(primary_metric)
        total_val = float(df[primary_metric].sum(skipna=True)) if not is_primary_rate else float(df[primary_metric].mean(skipna=True))
        sorted_df = df[[entity, primary_metric]].dropna().sort_values(primary_metric, ascending=False)

        # Top 10
        for _, row in sorted_df.head(10).iterrows():
            ent_name = str(row[entity])
            v = float(row[primary_metric])
            pct = (v / total_val * 100) if (total_val and not is_primary_rate) else v
            top_entities.append(EntityContribution(entity=ent_name, value=v, pct_share=pct))

        # Bottom 5 (excluding 0 or negative if mostly positive)
        pos_df = sorted_df[sorted_df[primary_metric] > 0]
        tail_df = pos_df.tail(5) if not pos_df.empty else sorted_df.tail(5)
        for _, row in tail_df.iterrows():
            ent_name = str(row[entity])
            v = float(row[primary_metric])
            pct = (v / total_val * 100) if (total_val and not is_primary_rate) else v
            bottom_entities.append(EntityContribution(entity=ent_name, value=v, pct_share=pct))

        # Pareto 80/20 computation (meaningful for additive metrics like revenue, stock, sales)
        if not is_primary_rate and total_val > 0 and len(sorted_df) >= 3:
            cumsum = sorted_df[primary_metric].cumsum()
            cutoff_idx = int((cumsum <= 0.80 * total_val).sum()) + 1
            cutoff_idx = min(cutoff_idx, len(sorted_df))
            pct_items = (cutoff_idx / len(sorted_df)) * 100
            pareto_text = f"Top {cutoff_idx} items ({pct_items:.0f}% of total) account for 80% of {primary_metric}."

    # Outliers & domain signals (e.g. high stock with 0 sales)
    outliers: list[dict] = []
    sold_col = next((c for c in df.columns if re.search(r"(sold|sales|orders|movement)", c, re.I)), None)
    stock_col = next((c for c in df.columns if re.search(r"(stock|hand|inventory|qty|quantity)", c, re.I)), None)

    if sold_col and stock_col and pd.api.types.is_numeric_dtype(df[sold_col]) and pd.api.types.is_numeric_dtype(df[stock_col]):
        dead_stock = df[(df[sold_col] == 0) & (df[stock_col] > 0)]
        for _, r in dead_stock.head(5).iterrows():
            outliers.append(
                {
                    "item": str(r[entity]),
                    "issue": "Zero movement / Dead stock",
                    "detail": f"{int(r[stock_col])} units on hand with 0 units sold.",
                    "severity": "high",
                }
            )

    # Statistical z-score outliers on primary metric
    if primary_metric in df.columns and len(df) >= 5:
        s = df[primary_metric].dropna()
        mean, std = s.mean(), s.std(ddof=0)
        if std > 0:
            z_scores = (s - mean) / std
            for idx in z_scores[z_scores >= 2.5].index[:5]:
                row = df.loc[idx]
                outliers.append(
                    {
                        "item": str(row[entity]),
                        "issue": f"Unusually high {primary_metric}",
                        "detail": f"{row[primary_metric]:,.1f} (z={z_scores[idx]:.1f})",
                        "severity": "medium",
                    }
                )

    # Group rollups
    group_rollups: dict[str, pd.DataFrame] = {}
    if grouping_cols:
        for g in grouping_cols:
            if g in df.columns and g != entity:
                agg_dict = {m: "sum" if not is_rate_metric(m) else "mean" for m in all_metrics if m in df.columns}
                if agg_dict:
                    grp = df.groupby(g).agg(agg_dict).reset_index()
                    group_rollups[g] = grp

    return SnapshotAnalyticsResult(
        total_items=total_items,
        kpis=kpis,
        top_entities=top_entities,
        bottom_entities=bottom_entities,
        pareto_text=pareto_text,
        outliers=outliers,
        group_rollups=group_rollups,
    )


def compute_project_analytics(
    df: pd.DataFrame,
    task_col: str | None,
    project_col: str | None,
    assignee_col: str | None,
    progress_col: str | None,
    duration_col: str | None,
    start_date_col: str | None,
    end_date_col: str | None,
) -> ProjectAnalyticsResult:
    """Compute project portfolio KPIs, team workload, and at-risk task signals."""
    total_tasks = len(df)

    # Normalise progress column to 0..100
    prog_series = pd.Series(0.0, index=df.index)
    if progress_col and progress_col in df.columns and pd.api.types.is_numeric_dtype(df[progress_col]):
        s = df[progress_col].fillna(0.0)
        # If stored as 0..1, scale to 0..100
        if s.max() <= 1.0 and s.max() > 0:
            prog_series = s * 100.0
        else:
            prog_series = s

    completed = int((prog_series >= 99.5).sum())
    unstarted = int((prog_series <= 0.5).sum())
    in_progress = total_tasks - completed - unstarted
    avg_progress = float(prog_series.mean()) if total_tasks else 0.0

    total_days = 0.0
    if duration_col and duration_col in df.columns and pd.api.types.is_numeric_dtype(df[duration_col]):
        total_days = float(df[duration_col].sum(skipna=True))
    elif start_date_col and end_date_col and start_date_col in df.columns and end_date_col in df.columns:
        s_date = pd.to_datetime(df[start_date_col], errors="coerce")
        e_date = pd.to_datetime(df[end_date_col], errors="coerce")
        dur = (e_date - s_date).dt.days.dropna()
        total_days = float(dur.sum()) if not dur.empty else 0.0

    kpis = {
        "Completion Rate": KPIResult("Completion Rate", "%", avg_progress, None, None, None, True),
        "Total Tasks": KPIResult("Total Tasks", None, float(total_tasks), None, None, None),
        "Completed Tasks": KPIResult("Completed Tasks", None, float(completed), None, None, None),
        "In-Progress Tasks": KPIResult("In-Progress Tasks", None, float(in_progress), None, None, None),
        "Unstarted Tasks": KPIResult("Unstarted Tasks", None, float(unstarted), None, None, None),
    }
    if total_days > 0:
        kpis["Total Days Required"] = KPIResult("Total Days Required", "days", total_days, None, None, None)

    # Project-level rollup
    proj_col = project_col or (task_col if task_col in df.columns else df.columns[0])
    project_df = pd.DataFrame()
    if proj_col in df.columns:
        df_proj = df.copy()
        df_proj["_prog_clean"] = prog_series
        agg_map = {"_prog_clean": "mean"}
        if duration_col and duration_col in df.columns:
            agg_map[duration_col] = "sum"
        project_df = df_proj.groupby(proj_col).agg(agg_map).reset_index()
        project_df = project_df.rename(columns={"_prog_clean": "Avg Progress (%)"})
        project_df["Tasks Count"] = df_proj.groupby(proj_col).size().values

    # Assignee-level workload rollup
    assignee_df = pd.DataFrame()
    if assignee_col and assignee_col in df.columns:
        df_asgn = df.copy()
        df_asgn["_prog_clean"] = prog_series
        agg_map = {"_prog_clean": "mean"}
        if duration_col and duration_col in df.columns:
            agg_map[duration_col] = "sum"
        assignee_df = df_asgn.groupby(assignee_col).agg(agg_map).reset_index()
        assignee_df = assignee_df.rename(columns={"_prog_clean": "Avg Progress (%)"})
        assignee_df["Tasks Assigned"] = df_asgn.groupby(assignee_col).size().values

    # At-risk tasks (0% progress with duration >= 20 days or end date in past)
    at_risk: list[dict] = []
    task_name_col = task_col or proj_col
    for idx, row in df.iterrows():
        prog = prog_series.loc[idx]
        task_name = str(row[task_name_col])
        dur = float(row[duration_col]) if (duration_col and duration_col in df.columns and pd.notna(row[duration_col])) else 0.0

        if prog <= 0.0 and dur >= 20:
            at_risk.append(
                {
                    "task": task_name,
                    "project": str(row.get(proj_col, "")),
                    "assignee": str(row.get(assignee_col, "")),
                    "reason": f"Long-duration task ({dur:.0f} days) not yet started (0%).",
                    "severity": "high",
                }
            )

    return ProjectAnalyticsResult(
        total_tasks=total_tasks,
        completed_tasks=completed,
        in_progress_tasks=in_progress,
        unstarted_tasks=unstarted,
        completion_rate_pct=avg_progress,
        total_days=total_days,
        kpis=kpis,
        project_rollups=project_df,
        assignee_workload=assignee_df,
        at_risk_tasks=at_risk,
    )
