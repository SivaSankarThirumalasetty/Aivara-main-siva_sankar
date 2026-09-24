"""
ui/drivers_view.py — Operations-equivalent view: ranked dimension drivers
across all classified dimension columns, not just the top one.
"""

from __future__ import annotations

import streamlit as st

from ui.components import build_drivers_figure


def render_drivers_view(pipeline_result: dict, key_prefix: str = "drivers") -> None:
    st.header("Operations")
    st.caption("Which dimensions explain the change, ranked by how concentrated their contribution is.")

    all_drivers = pipeline_result.get("all_dimension_drivers", [])
    if not all_drivers:
        st.info("No dimension columns available to decompose the change by.")
        return

    for driver in all_drivers:
        with st.expander(f"{driver.dimension} (concentration score: {driver.concentration_score:.2f})", expanded=(driver is all_drivers[0])):
            if driver.contributions:
                fig = build_drivers_figure(driver.top_positive + driver.top_negative, driver.dimension)
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_{driver.dimension}")

            st.write("**Top positive contributors**")
            for c in driver.top_positive:
                st.write(f"- {c.category}: {c.contribution_pct:+.1f}%")

            if driver.top_negative:
                st.write("**Top negative contributors**")
                for c in driver.top_negative:
                    st.write(f"- {c.category}: {c.contribution_pct:+.1f}%")
