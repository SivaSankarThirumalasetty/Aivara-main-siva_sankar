"""
ui/risk_view.py — Risk tab: flagged anomalies / risk signals.
"""

from __future__ import annotations

import streamlit as st

SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def render_risk_view(pipeline_result: dict) -> None:
    st.header("Risk")

    signals = pipeline_result.get("risk_signals", [])
    st.metric("Risk signals", len(signals))

    if not signals:
        st.success("No risk signals flagged for this dataset.")
        return

    for s in signals:
        icon = SEVERITY_ICON.get(s.severity, "⚪")
        st.write(f"{icon} **{s.severity.upper()}** — {s.description}")


def render_forecast_stub() -> None:
    st.header("Forecast")
    st.info("Forecasting is not available in this version — coming in a future release.")
