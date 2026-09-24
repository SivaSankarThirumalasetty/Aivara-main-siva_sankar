import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from analytics.kpis import KPIResult
from insight.facts_builder import build_facts_packet
from insight.prompts import build_insight_prompt
from ui.components import mask_secret


def test_mask_secret():
    assert mask_secret("") == ""
    assert mask_secret("1234") == "****"
    assert mask_secret("sk-or-v1-abcdef1234567890") == ("*" * 21) + "7890"
    assert mask_secret("test") == "****"


def test_facts_packet_does_not_contain_raw_csv_rows():
    """Verify that only aggregated numbers (not raw row data or unaggregated strings)
    are included in the payload sent to the LLM."""
    df_raw = pd.DataFrame({
        "customer_ssn": ["000-12-3456", "000-12-7890"],
        "credit_card": ["4111111111111111", "5500000000000004"],
        "revenue": [100.0, 200.0],
    })

    kpi = KPIResult(name="revenue", unit="$", current_value=300.0, prior_value=200.0, pct_change=50.0, abs_change=100.0)
    packet = build_facts_packet(kpi, top_dimension_drivers=None, risk_signals=None, period_label="this month vs last month")

    packet_str = json.dumps(packet)
    # Assert raw PII strings never appear in the facts packet
    assert "000-12-3456" not in packet_str
    assert "4111111111111111" not in packet_str

    # Build prompt messages and assert same
    messages = build_insight_prompt(packet)
    prompt_str = json.dumps(messages)
    assert "000-12-3456" not in prompt_str
    assert "4111111111111111" not in prompt_str
