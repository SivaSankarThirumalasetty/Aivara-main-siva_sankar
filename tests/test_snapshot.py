import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from analytics.snapshot import compute_snapshot_analytics, compute_project_analytics


def test_compute_snapshot_analytics_basic():
    df = pd.DataFrame(
        {
            "product": [f"Item {i}" for i in range(20)],
            "category": (["Cat A", "Cat B"] * 10),
            "units_sold": [10, 20, 0, 5, 50, 100] + [10] * 14,
            "stock": [50, 40, 30, 20, 10, 5] + [15] * 14,
        }
    )
    res = compute_snapshot_analytics(
        df=df,
        entity_col="product",
        primary_metric="units_sold",
        secondary_metrics=["stock"],
        grouping_cols=["category"],
    )
    assert res.total_items == 20
    assert "units_sold" in res.kpis
    assert len(res.top_entities) <= 10
    assert res.top_entities[0].entity == "Item 5"  # highest units sold (100)
    assert "category" in res.group_rollups
    # Item 2 has 0 sold and 30 in stock -> should be flagged as dead stock
    assert any("Dead stock" in str(o.get("issue")) for o in res.outliers)


def test_compute_project_analytics_basic():
    df = pd.DataFrame(
        {
            "task": ["Research", "Design", "Dev", "QA"],
            "project": ["App", "App", "App", "App"],
            "assignee": ["Alice", "Bob", "Charlie", "Alice"],
            "days": [10, 15, 25, 5],
            "progress": [1.0, 0.5, 0.0, 0.0],  # 1 completed, 1 in-progress, 2 unstarted
        }
    )
    res = compute_project_analytics(
        df=df,
        task_col="task",
        project_col="project",
        assignee_col="assignee",
        progress_col="progress",
        duration_col="days",
        start_date_col=None,
        end_date_col=None,
    )
    assert res.total_tasks == 4
    assert res.completed_tasks == 1
    assert res.in_progress_tasks == 1
    assert res.unstarted_tasks == 2
    assert res.completion_rate_pct == 37.5  # (100 + 50 + 0 + 0) / 4
    assert res.total_days == 55.0
    assert len(res.assignee_workload) == 3  # Alice, Bob, Charlie
    # Dev has 0% progress and 25 days duration -> at risk
    assert any(t["task"] == "Dev" for t in res.at_risk_tasks)
