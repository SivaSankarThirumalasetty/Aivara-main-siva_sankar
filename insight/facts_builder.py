"""
insight/facts_builder.py — turns local analytics output into a compact JSON
facts packet. This is the ONLY thing that ever leaves the machine and goes
to the LLM (Section 9.2). No raw rows, ever.
"""

from __future__ import annotations

import math
from typing import Any

from analytics.anomalies import RiskSignal
from analytics.drivers import DimensionDriverResult
from analytics.kpis import KPIResult


def build_facts_packet(
    kpi: KPIResult,
    top_dimension_drivers: DimensionDriverResult | None,
    risk_signals: list[RiskSignal] | None,
    period_label: str,
    unit: str | None = None,
) -> dict[str, Any]:
    """Builds the small JSON object described in Section 9.2 for a single
    KPI's insight call. Only aggregated numbers are included — never raw
    records, never other rows/columns from the source data."""

    packet: dict[str, Any] = {
        "kpi_name": kpi.name,
        "unit": unit or kpi.unit,
        "current_value": round(kpi.current_value, 2),
        "prior_value": round(kpi.prior_value, 2) if kpi.prior_value is not None else None,
        "pct_change": (
            round(kpi.pct_change, 2)
            if (kpi.pct_change is not None and not math.isinf(kpi.pct_change))
            else None
        ),
        "top_positive_drivers": [],
        "top_negative_drivers": [],
        "flagged_risk_items": [],
        "period_label": period_label,
    }

    if top_dimension_drivers is not None:
        packet["top_positive_drivers"] = [
            {
                "dimension": top_dimension_drivers.dimension,
                "category": c.category,
                "contribution_pct": round(c.contribution_pct, 1),
            }
            for c in top_dimension_drivers.top_positive
        ]
        packet["top_negative_drivers"] = [
            {
                "dimension": top_dimension_drivers.dimension,
                "category": c.category,
                "contribution_pct": round(c.contribution_pct, 1),
            }
            for c in top_dimension_drivers.top_negative
        ]

    if risk_signals:
        packet["flagged_risk_items"] = [
            {"description": s.description, "severity": s.severity} for s in risk_signals[:5]
        ]

    return packet


def build_snapshot_facts_packet(
    total_items: int,
    entity_dimension: str | None,
    primary_metric: str | None,
    primary_metric_value: float,
    top_entity: str | None,
    top_entity_value: float | None,
    top_entity_pct: float | None,
    pareto_text: str | None = None,
    risk_items: list[str] | None = None,
) -> dict[str, Any]:
    """Compact aggregated facts packet for cross-sectional snapshot datasets."""
    return {
        "analysis_type": "snapshot",
        "total_items": total_items,
        "entity_dimension": entity_dimension or "items",
        "primary_metric": primary_metric,
        "total_value": round(primary_metric_value, 2),
        "top_performer": top_entity,
        "top_performer_value": round(top_entity_value, 2) if top_entity_value is not None else None,
        "top_performer_share_pct": round(top_entity_pct, 1) if top_entity_pct is not None else None,
        "concentration_insight": pareto_text,
        "flagged_risk_items": risk_items or [],
    }


def build_project_facts_packet(
    total_tasks: int,
    completion_rate_pct: float,
    completed_tasks: int,
    unstarted_tasks: int,
    total_days: float,
    at_risk_count: int,
) -> dict[str, Any]:
    """Compact aggregated facts packet for project / task portfolio datasets."""
    return {
        "analysis_type": "project_portfolio",
        "total_tasks": total_tasks,
        "overall_completion_rate": f"{completion_rate_pct:.1f}%",
        "completed_tasks_count": completed_tasks,
        "unstarted_tasks_count": unstarted_tasks,
        "total_duration_days": total_days,
        "at_risk_tasks_count": at_risk_count,
    }


def reduce_facts_packet_if_needed(facts_packet: dict, max_tokens: int = 2500) -> dict:
    """If facts packet exceeds max_tokens estimate, safely reduce lists and string lengths
    locally without losing key primary metrics."""
    import json
    from insight.budget import estimate_tokens

    packet_str = json.dumps(facts_packet, ensure_ascii=False)
    if estimate_tokens(packet_str) <= max_tokens:
        return facts_packet

    reduced = dict(facts_packet)
    if "top_positive_drivers" in reduced and isinstance(reduced["top_positive_drivers"], list):
        reduced["top_positive_drivers"] = [
            {**d, "category": str(d.get("category", ""))[:40]}
            for d in reduced["top_positive_drivers"][:2]
            if isinstance(d, dict)
        ]
    if "top_negative_drivers" in reduced and isinstance(reduced["top_negative_drivers"], list):
        reduced["top_negative_drivers"] = [
            {**d, "category": str(d.get("category", ""))[:40]}
            for d in reduced["top_negative_drivers"][:1]
            if isinstance(d, dict)
        ]
    if "flagged_risk_items" in reduced and isinstance(reduced["flagged_risk_items"], list):
        reduced["flagged_risk_items"] = [
            {**s, "description": str(s.get("description", ""))[:80]}
            for s in reduced["flagged_risk_items"][:2]
            if isinstance(s, dict)
        ]

    return reduced

