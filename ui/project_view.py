"""
ui/project_view.py — Project & Task Portfolio Dashboard.

Rendered when a dataset contains project/task timelines, milestone dates,
task progress, and team assignments.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from analytics.profiler import DatasetProfile
from analytics.snapshot import ProjectAnalyticsResult
from ui.components import humanize_col, render_kpi_tile, render_insight_card


def render_project_overview_view(
    df: pd.DataFrame,
    project_res: ProjectAnalyticsResult,
    profile: DatasetProfile,
    model: str,
    allow_paid: bool = False,
) -> None:
    st.header("Portfolio Overview")
    st.caption(f"Status and velocity across {project_res.total_tasks} tasks.")

    # Top KPI tiles
    kpis = list(project_res.kpis.values())
    if kpis:
        cols = st.columns(min(5, len(kpis)))
        for i, kpi in enumerate(kpis[:5]):
            with cols[i % len(cols)]:
                render_kpi_tile(kpi)

    # Visual portfolio progress bar
    st.progress(
        min(1.0, max(0.0, project_res.completion_rate_pct / 100.0)),
        text=f"Overall Portfolio Completion: {project_res.completion_rate_pct:.1f}%",
    )

    # Project-level Progress Bar Chart
    if not project_res.project_rollups.empty:
        proj_col = project_res.project_rollups.columns[0]
        h_proj = humanize_col(proj_col)
        fig = px.bar(
            project_res.project_rollups.sort_values("Avg Progress (%)", ascending=False),
            x=proj_col,
            y="Avg Progress (%)",
            labels={proj_col: h_proj, "Avg Progress (%)": "Average Progress (%)"},
            title=f"Average Completion by {h_proj}",
            template="plotly_white",
            height=420,
            color="Avg Progress (%)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width="stretch", key="project_overview_chart")
        st.session_state.setdefault("_export_figures", {})["Executive_trend"] = fig

    # Insight Card
    facts_packet = {
        "dataset_type": "project",
        "total_tasks": project_res.total_tasks,
        "completed_tasks": project_res.completed_tasks,
        "unstarted_tasks": project_res.unstarted_tasks,
        "completion_rate_pct": f"{project_res.completion_rate_pct:.1f}%",
        "total_days": project_res.total_days,
        "at_risk_count": len(project_res.at_risk_tasks),
    }
    render_insight_card("Executive", facts_packet, model, allow_paid=allow_paid)


def render_project_gantt_view(
    df: pd.DataFrame,
    project_res: ProjectAnalyticsResult,
    profile: DatasetProfile,
) -> None:
    st.header("Timeline & Schedule (Gantt)")
    st.caption("Visual schedule of tasks from start date to target completion.")

    task_col = profile.entity_dimension or df.columns[0]
    start_col = profile.primary_date_col
    end_col = profile.end_date_col

    if not start_col or not end_col:
        st.info("Start date and End date columns are required to render the timeline Gantt chart.")
        return

    # Ensure datetime format for plotly timeline
    plot_df = df.copy()
    plot_df[start_col] = pd.to_datetime(plot_df[start_col], errors="coerce")
    plot_df[end_col] = pd.to_datetime(plot_df[end_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[start_col, end_col])

    if plot_df.empty:
        st.warning("No valid start and end dates found to plot timeline.")
        return

    # Sort by start date
    plot_df = plot_df.sort_values(start_col)

    color_col = next((c for c in profile.grouping_dimensions if c in df.columns), None)
    prog_col = next((m for m in profile.rate_metrics if m in df.columns), None)

    try:
        fig = px.timeline(
            plot_df,
            x_start=start_col,
            x_end=end_col,
            y=task_col,
            color=color_col if color_col else (prog_col if prog_col else None),
            title="Task Timeline Schedule",
            labels={task_col: humanize_col(task_col), start_col: "Start Date", end_col: "End Date"},
            template="plotly_white",
            height=max(450, len(plot_df) * 20),
        )
        fig.update_yaxes(autorange="reversed")  # First task at the top
        st.plotly_chart(fig, width="stretch", key="project_gantt_chart")
        st.session_state.setdefault("_export_figures", {})["Project_gantt"] = fig
    except Exception as exc:
        st.warning(f"Could not render Gantt chart: {exc}")


def render_project_team_view(
    df: pd.DataFrame,
    project_res: ProjectAnalyticsResult,
    profile: DatasetProfile,
) -> None:
    st.header("Team & Workload Distribution")
    st.caption("Workload balance across assignees, task volume, and individual velocity.")

    if project_res.assignee_workload.empty:
        st.info("No assignee / team dimension detected to decompose workload by.")
        return

    asgn_col = project_res.assignee_workload.columns[0]
    h_asgn = humanize_col(asgn_col)

    c1, c2 = st.columns(2)
    with c1:
        fig_tasks = px.bar(
            project_res.assignee_workload.sort_values("Tasks Assigned", ascending=False),
            x=asgn_col,
            y="Tasks Assigned",
            title=f"Tasks Assigned per {h_asgn}",
            labels={asgn_col: h_asgn},
            template="plotly_white",
            color="Tasks Assigned",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_tasks, width="stretch", key="project_workload_tasks_chart")
        st.session_state.setdefault("_export_figures", {})["Project_workload"] = fig_tasks

    with c2:
        fig_prog = px.bar(
            project_res.assignee_workload.sort_values("Avg Progress (%)", ascending=False),
            x=asgn_col,
            y="Avg Progress (%)",
            title=f"Average Completion Rate (%) per {h_asgn}",
            labels={asgn_col: h_asgn},
            template="plotly_white",
            color="Avg Progress (%)",
            color_continuous_scale="Greens",
        )
        st.plotly_chart(fig_prog, width="stretch", key="project_workload_prog_chart")

    with st.expander("📋 Assignee Workload Breakdown Table", expanded=False):
        st.dataframe(project_res.assignee_workload.reset_index(drop=True))


def render_project_at_risk_view(project_res: ProjectAnalyticsResult) -> None:
    st.header("At-Risk & Stalled Tasks")
    st.caption("Tasks with zero progress despite significant required duration.")

    if not project_res.at_risk_tasks:
        st.info("✅ No critical delays or high-duration unstarted tasks flagged.")
        return

    for t in project_res.at_risk_tasks:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"⚠️ **{t['task']}**")
                st.caption(f"Project: `{t.get('project', 'N/A')}` | Assigned To: `{t.get('assignee', 'N/A')}`")
                st.write(t["reason"])
            with c2:
                st.caption("Status: `UNSTARTED`")
