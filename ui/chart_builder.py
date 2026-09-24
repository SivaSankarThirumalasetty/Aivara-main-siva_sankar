"""
ui/chart_builder.py — Interactive Chart Explorer widget.

Lets the user pick chart type, X/Y columns, and an optional color/group
dimension directly from the uploaded dataset's column headers. Supports 10
chart types via Plotly. All axis labels use humanized column names so charts
read naturally regardless of whether the source was a CSV (snake_case) or
Excel file (already spaced).

Session-state keys (all prefixed "chartexp_") persist the user's choices
across Streamlit rerenders without resetting on every interaction.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ui.components import humanize_col

# ------------------------------------------------------------------ constants

CHART_TYPES = [
    "Line",
    "Area",
    "Bar",
    "Horizontal Bar",
    "Scatter",
    "Pie",
    "Donut",
    "Box",
    "Histogram",
    "Heatmap",
    "Timeline (Gantt)",
]

# Chart types that require BOTH an X and a Y column
_NEEDS_XY = {"Line", "Area", "Bar", "Horizontal Bar", "Scatter", "Box"}
# Chart types that only need a single value column (+ optional label col)
_SINGLE_VAR = {"Histogram"}
# Chart types that use label + value (no time axis)
_LABEL_VALUE = {"Pie", "Donut"}
# Heatmap is its own special case
_HEATMAP = {"Heatmap"}
_TIMELINE = {"Timeline (Gantt)"}


# ----------------------------------------------------------------------- util

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def _all_cols(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def _default_x(df: pd.DataFrame, chart_type: str) -> str:
    """Pick a sensible default X column based on chart type."""
    if chart_type in _LABEL_VALUE:
        cats = _categorical_cols(df)
        return cats[0] if cats else df.columns[0]
    # prefer a date/time-like column for XY charts
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # fall back to first non-numeric, then first column
    cats = _categorical_cols(df)
    return cats[0] if cats else df.columns[0]


def _default_y(df: pd.DataFrame) -> str:
    nums = _numeric_cols(df)
    return nums[0] if nums else df.columns[-1]


# ----------------------------------------------------------------- main build

def build_chart_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None,
    y_col: str | None,
    color_col: str | None = None,
    agg: str = "Sum",
) -> go.Figure | None:
    """Build a Plotly Figure for `chart_type` given the selected columns.

    Returns None if the required columns are missing/invalid.
    All axis titles use humanized column names.
    """
    if df.empty:
        return None

    h_x = humanize_col(x_col) if x_col else ""
    h_y = humanize_col(y_col) if y_col else ""
    h_color = humanize_col(color_col) if color_col else None

    # ---- aggregate Y by X (and optionally color) for most chart types ----
    def _aggregate(grp_cols: list[str]) -> pd.DataFrame:
        valid = [c for c in grp_cols if c in df.columns]
        if not valid or y_col not in df.columns:
            return df.copy()
        agg_fn = "mean" if agg == "Mean" else "sum"
        return df.groupby(valid)[y_col].agg(agg_fn).reset_index()

    try:
        if chart_type in ("Line", "Area"):
            if not x_col or not y_col:
                return None
            grp_cols = [x_col] + ([color_col] if color_col else [])
            plot_df = _aggregate(grp_cols)
            plot_df = plot_df.sort_values(x_col)
            if color_col and color_col in plot_df.columns:
                fig = px.line(
                    plot_df, x=x_col, y=y_col, color=color_col,
                    labels={x_col: h_x, y_col: h_y, color_col: h_color},
                    title=f"{h_y} by {h_x}",
                )
            else:
                fig = px.line(
                    plot_df, x=x_col, y=y_col,
                    labels={x_col: h_x, y_col: h_y},
                    title=f"{h_y} over {h_x}",
                )
            if chart_type == "Area":
                for trace in fig.data:
                    trace.fill = "tozeroy"
                    trace.fillcolor = "rgba(26,115,232,0.15)"

        elif chart_type == "Bar":
            if not x_col or not y_col:
                return None
            grp_cols = [x_col] + ([color_col] if color_col else [])
            plot_df = _aggregate(grp_cols)
            if color_col and color_col in plot_df.columns:
                fig = px.bar(
                    plot_df, x=x_col, y=y_col, color=color_col, barmode="group",
                    labels={x_col: h_x, y_col: h_y, color_col: h_color},
                    title=f"{h_y} by {h_x}",
                )
            else:
                fig = px.bar(
                    plot_df, x=x_col, y=y_col,
                    labels={x_col: h_x, y_col: h_y},
                    title=f"{h_y} by {h_x}",
                )

        elif chart_type == "Horizontal Bar":
            if not x_col or not y_col:
                return None
            grp_cols = [x_col] + ([color_col] if color_col else [])
            plot_df = _aggregate(grp_cols)
            plot_df = plot_df.sort_values(y_col, ascending=True)
            if color_col and color_col in plot_df.columns:
                fig = px.bar(
                    plot_df, x=y_col, y=x_col, color=color_col,
                    orientation="h", barmode="group",
                    labels={x_col: h_x, y_col: h_y, color_col: h_color},
                    title=f"{h_y} by {h_x}",
                )
            else:
                fig = px.bar(
                    plot_df, x=y_col, y=x_col, orientation="h",
                    labels={x_col: h_x, y_col: h_y},
                    title=f"{h_y} by {h_x}",
                )

        elif chart_type == "Scatter":
            if not x_col or not y_col:
                return None
            scatter_df = df[[c for c in [x_col, y_col, color_col] if c and c in df.columns]].dropna()
            if color_col and color_col in scatter_df.columns:
                fig = px.scatter(
                    scatter_df, x=x_col, y=y_col, color=color_col,
                    labels={x_col: h_x, y_col: h_y, color_col: h_color},
                    title=f"{h_y} vs {h_x}",
                )
            else:
                fig = px.scatter(
                    scatter_df, x=x_col, y=y_col,
                    labels={x_col: h_x, y_col: h_y},
                    title=f"{h_y} vs {h_x}",
                )

        elif chart_type in ("Pie", "Donut"):
            if not x_col or not y_col:
                return None
            grp_df = df.groupby(x_col)[y_col].sum().reset_index()
            grp_df.columns = ["label", "value"]
            fig = go.Figure(
                go.Pie(
                    labels=grp_df["label"],
                    values=grp_df["value"],
                    hole=0.4 if chart_type == "Donut" else 0.0,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(title=f"{h_y} by {h_x}")

        elif chart_type == "Box":
            if not y_col:
                return None
            box_cols = [c for c in [x_col, y_col, color_col] if c and c in df.columns]
            box_df = df[box_cols].dropna(subset=[y_col])
            if x_col and x_col in box_df.columns:
                fig = px.box(
                    box_df, x=x_col, y=y_col, color=color_col if color_col else None,
                    labels={x_col: h_x, y_col: h_y},
                    title=f"Distribution of {h_y}" + (f" by {h_x}" if x_col else ""),
                )
            else:
                fig = px.box(
                    box_df, y=y_col,
                    labels={y_col: h_y},
                    title=f"Distribution of {h_y}",
                )

        elif chart_type == "Histogram":
            if not x_col:
                return None
            hist_df = df[[x_col]].dropna() if x_col in df.columns else df
            fig = px.histogram(
                hist_df, x=x_col,
                labels={x_col: h_x},
                title=f"Frequency Distribution of {h_x}",
                nbins=30,
            )

        elif chart_type == "Heatmap":
            nums = _numeric_cols(df)
            if len(nums) < 2:
                st.info("Heatmap requires at least 2 numeric columns.")
                return None
            corr = df[nums].corr()
            h_labels = [humanize_col(c) for c in corr.columns]
            fig = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=h_labels,
                    y=h_labels,
                    colorscale="RdBu",
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                    showscale=True,
                )
            )
            fig.update_layout(
                title="Correlation Heatmap",
                margin=dict(l=80, r=20, t=60, b=80),
            )

        elif chart_type == "Timeline (Gantt)":
            if not x_col or not y_col:
                return None
            gantt_df = df.copy()
            gantt_df[x_col] = pd.to_datetime(gantt_df[x_col], errors="coerce")
            gantt_df[y_col] = pd.to_datetime(gantt_df[y_col], errors="coerce")
            gantt_df = gantt_df.dropna(subset=[x_col, y_col])
            if gantt_df.empty:
                return None
            task_col = color_col if (color_col and color_col in gantt_df.columns) else gantt_df.columns[0]
            fig = px.timeline(
                gantt_df,
                x_start=x_col,
                x_end=y_col,
                y=task_col,
                labels={task_col: humanize_col(task_col), x_col: h_x, y_col: h_y},
                title=f"Timeline from {h_x} to {h_y}",
            )
            fig.update_yaxes(autorange="reversed")

        else:
            return None

        # common layout polish
        if chart_type not in ("Heatmap",):
            fig.update_layout(
                template="plotly_white",
                height=480,
                margin=dict(l=50, r=20, t=60, b=50),
            )

        return fig

    except Exception as exc:  # noqa: BLE001 — never crash the dashboard
        st.warning(f"Could not render chart: {exc}")
        return None


# ------------------------------------------------------------------ UI widget

def render_chart_explorer(df: pd.DataFrame, key_prefix: str = "chartexp") -> None:
    """Render the full interactive Chart Explorer widget.

    Reads column headers directly from `df`, persists user selections in
    session_state, and renders the chart with Plotly.
    """
    st.header("Charts")
    st.caption(
        "Explore your data with any chart type. Axis labels come directly "
        "from your file's column headers."
    )

    if df.empty or len(df.columns) < 1:
        st.info("Upload a dataset first to explore charts.")
        return

    all_cols = _all_cols(df)
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    # --- Control row 1: chart type + aggregation --------------------------
    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        chart_type = st.selectbox(
            "Chart type",
            CHART_TYPES,
            index=CHART_TYPES.index(
                st.session_state.get(f"{key_prefix}_chart_type", "Bar")
            ),
            key=f"{key_prefix}_chart_type",
            help="Select how to visualize your data.",
        )
    with ctrl2:
        agg = st.selectbox(
            "Aggregation",
            ["Sum", "Mean"],
            index=["Sum", "Mean"].index(
                st.session_state.get(f"{key_prefix}_agg", "Sum")
            ),
            key=f"{key_prefix}_agg",
            help="How to aggregate Y values when multiple rows share the same X value.",
            disabled=chart_type in _SINGLE_VAR | _HEATMAP,
        )

    # --- Dynamic column pickers depending on chart type -------------------
    x_col: str | None = None
    y_col: str | None = None
    color_col: str | None = None

    if chart_type in _HEATMAP:
        st.caption("Heatmap shows correlations between all numeric columns automatically.")

    elif chart_type in _SINGLE_VAR:
        # Histogram only needs one column
        default_x = st.session_state.get(f"{key_prefix}_x_col") or (
            num_cols[0] if num_cols else all_cols[0]
        )
        if default_x not in all_cols:
            default_x = all_cols[0]
        x_col = st.selectbox(
            "Column to plot",
            all_cols,
            index=all_cols.index(default_x),
            key=f"{key_prefix}_x_col",
            format_func=humanize_col,
        )

    elif chart_type in _LABEL_VALUE:
        # Pie / Donut: label column + value column
        c1, c2 = st.columns(2)
        default_x = st.session_state.get(f"{key_prefix}_x_col") or _default_x(df, chart_type)
        default_y = st.session_state.get(f"{key_prefix}_y_col") or _default_y(df)
        if default_x not in all_cols:
            default_x = all_cols[0]
        if default_y not in all_cols:
            default_y = all_cols[0]
        with c1:
            x_col = st.selectbox(
                "Label column (slices)",
                all_cols,
                index=all_cols.index(default_x),
                key=f"{key_prefix}_x_col",
                format_func=humanize_col,
            )
        with c2:
            y_col = st.selectbox(
                "Value column (size)",
                num_cols if num_cols else all_cols,
                index=(num_cols if num_cols else all_cols).index(default_y)
                if default_y in (num_cols if num_cols else all_cols)
                else 0,
                key=f"{key_prefix}_y_col",
                format_func=humanize_col,
            )

    elif chart_type in _TIMELINE:
        c1, c2, c3 = st.columns(3)
        date_candidates = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower()]
        if len(date_candidates) >= 2:
            def_start, def_end = date_candidates[0], date_candidates[1]
        elif len(date_candidates) == 1:
            def_start = date_candidates[0]
            def_end = all_cols[1] if len(all_cols) > 1 else all_cols[0]
        else:
            def_start = all_cols[0]
            def_end = all_cols[1] if len(all_cols) > 1 else all_cols[0]

        with c1:
            x_col = st.selectbox(
                "Start Date column",
                all_cols,
                index=all_cols.index(def_start) if def_start in all_cols else 0,
                key=f"{key_prefix}_x_col",
                format_func=humanize_col,
            )
        with c2:
            y_col = st.selectbox(
                "End Date column",
                all_cols,
                index=all_cols.index(def_end) if def_end in all_cols else 0,
                key=f"{key_prefix}_y_col",
                format_func=humanize_col,
            )
        with c3:
            label_opts = [c for c in all_cols if c not in (x_col, y_col)] or all_cols
            color_col = st.selectbox(
                "Task / Entity column",
                label_opts,
                index=0,
                key=f"{key_prefix}_color_col",
                format_func=humanize_col,
            )

    else:
        # Standard XY charts
        c1, c2, c3 = st.columns(3)
        default_x = st.session_state.get(f"{key_prefix}_x_col") or _default_x(df, chart_type)
        default_y = st.session_state.get(f"{key_prefix}_y_col") or _default_y(df)
        if default_x not in all_cols:
            default_x = all_cols[0]
        if default_y not in (num_cols if num_cols else all_cols):
            default_y = (num_cols if num_cols else all_cols)[0]

        x_label = "X axis (category / time)" if chart_type != "Histogram" else "Column"
        y_label = "Y axis (value)"

        with c1:
            x_col = st.selectbox(
                x_label,
                all_cols,
                index=all_cols.index(default_x),
                key=f"{key_prefix}_x_col",
                format_func=humanize_col,
            )
        with c2:
            y_options = num_cols if num_cols else all_cols
            y_col = st.selectbox(
                y_label,
                y_options,
                index=y_options.index(default_y) if default_y in y_options else 0,
                key=f"{key_prefix}_y_col",
                format_func=humanize_col,
            )
        with c3:
            color_options = ["(none)"] + [c for c in all_cols if c not in (x_col, y_col)]
            saved_color = st.session_state.get(f"{key_prefix}_color_col", "(none)")
            if saved_color not in color_options:
                saved_color = "(none)"
            color_sel = st.selectbox(
                "Color / group by",
                color_options,
                index=color_options.index(saved_color),
                key=f"{key_prefix}_color_col",
                format_func=lambda c: humanize_col(c) if c != "(none)" else "None",
            )
            color_col = color_sel if color_sel != "(none)" else None

    # --- Render the figure ------------------------------------------------
    st.divider()
    fig = build_chart_figure(
        df,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        agg=agg,
    )
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_main_chart")

    # --- Optional data preview --------------------------------------------
    with st.expander("📋 Raw data preview", expanded=False):
        preview_cols = [c for c in [x_col, y_col, color_col] if c and c in df.columns]
        if preview_cols:
            st.dataframe(df[preview_cols].head(100))
        else:
            st.dataframe(df.head(100))
