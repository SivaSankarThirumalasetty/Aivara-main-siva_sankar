import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from analytics.kpis import KPIResult
from insight.client import (
    BudgetExhaustedError,
    InsightResult,
    OpenRouterError,
    _split_into_sentences,
    generate_insight,
)
from insight.facts_builder import build_facts_packet
from insight import budget, cache as insight_cache


def test_split_into_sentences_handles_decimals_and_percentages():
    text = (
        "Revenue rose to $30,000.50 this week. "
        "North led with 81.8% of the gain. "
        "Investigate South's 9.1% decline before next review."
    )
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Revenue rose to $30,000.50 this week."
    assert sentences[1] == "North led with 81.8% of the gain."
    assert sentences[2] == "Investigate South's 9.1% decline before next review."


def test_split_into_sentences_handles_numbered_list():
    text = (
        "1) Revenue rose by 15.5% this week.\n"
        "2) Enterprise segment drove 90.2% of the expansion.\n"
        "3) Prioritize mid-market retention calls."
    )
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3
    assert "15.5%" in sentences[0]
    assert "90.2%" in sentences[1]
    assert "retention" in sentences[2]


def test_split_into_sentences_handles_structured_json():
    json_str = json.dumps({
        "headline": "Revenue grew $45,200.50 (+12.4%) this quarter.",
        "driver_explanation": "Direct sales channel contributed 78.5% of gross additions.",
        "suggested_action": "Increase inventory for top 3 SKUs.",
    })
    sentences = _split_into_sentences(json_str)
    assert len(sentences) == 3
    assert sentences[0] == "Revenue grew $45,200.50 (+12.4%) this quarter."
    assert sentences[1] == "Direct sales channel contributed 78.5% of gross additions."
    assert sentences[2] == "Increase inventory for top 3 SKUs."


def test_split_into_sentences_handles_markdown_fenced_json():
    fenced = """```json
{
  "headline": "Conversion rate improved by 3.2 percentage points.",
  "driver_explanation": "Mobile checkout revamp drove 85% of gains.",
  "suggested_action": "Roll out mobile checkout to EU region."
}
```"""
    sentences = _split_into_sentences(fenced)
    assert len(sentences) == 3
    assert "Conversion rate improved" in sentences[0]
    assert "Mobile checkout" in sentences[1]
    assert "EU region" in sentences[2]


def test_facts_builder_excludes_both_pos_and_neg_inf():
    kpi_pos = KPIResult("metric1", None, 100.0, 0.0, float("inf"), 100.0)
    packet_pos = build_facts_packet(kpi_pos, None, None, "test")
    assert packet_pos["pct_change"] is None

    kpi_neg = KPIResult("metric2", None, -50.0, 0.0, float("-inf"), -50.0)
    packet_neg = build_facts_packet(kpi_neg, None, None, "test")
    assert packet_neg["pct_change"] is None


def test_oversized_facts_packet_rejected_before_network_call(monkeypatch):
    called = {"hit": False}

    def _fail_if_called(*args, **kwargs):
        called["hit"] = True
        return "raw", "model"

    monkeypatch.setattr("insight.client._call_openrouter", _fail_if_called)

    # Giant facts packet that exceeds 3000 estimated tokens (4000 words = ~4014 tokens)
    giant_packet = {"kpi_name": "rev", "bloat": "word " * 4000}
    with pytest.raises(OpenRouterError, match="exceeds safe token budget"):
        generate_insight(giant_packet)
    assert not called["hit"]
