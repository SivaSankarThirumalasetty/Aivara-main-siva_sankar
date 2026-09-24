import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.anomalies import RiskSignal
from analytics.drivers import CategoryContribution, DimensionDriverResult
from analytics.kpis import KPIResult
from insight import budget, cache as insight_cache
from insight.client import (
    BudgetExhaustedError,
    InsightResult,
    OpenRouterError,
    _parse_and_validate_insight_json,
    generate_insight,
)
from insight.facts_builder import (
    build_facts_packet,
    reduce_facts_packet_if_needed,
)
from insight.prompts import cache_key_material


def test_valid_json_parsing():
    valid_payload = json.dumps({
        "headline": "Revenue grew by $15,200.50 (+8.5%) this quarter.",
        "driver_explanation": "North region contributed 65.2% of total gain.",
        "suggested_action": "Expand inventory in North region hubs.",
    })
    res = _parse_and_validate_insight_json(valid_payload)
    assert res["headline"] == "Revenue grew by $15,200.50 (+8.5%) this quarter."
    assert res["driver_explanation"] == "North region contributed 65.2% of total gain."
    assert res["suggested_action"] == "Expand inventory in North region hubs."


def test_valid_markdown_fenced_json():
    fenced_payload = """```json
{
  "headline": "Conversion rate increased to 4.25% (up 0.5pp).",
  "driver_explanation": "Mobile checkout overhaul drove 90% of gains.",
  "suggested_action": "A/B test new mobile checkout in EMEA."
}
```"""
    res = _parse_and_validate_insight_json(fenced_payload)
    assert res["headline"] == "Conversion rate increased to 4.25% (up 0.5pp)."
    assert res["driver_explanation"] == "Mobile checkout overhaul drove 90% of gains."
    assert res["suggested_action"] == "A/B test new mobile checkout in EMEA."


def test_valid_json_with_surrounding_preamble_and_unicode():
    text = """Here is the business analysis insight:
    {
      "headline": "Gross margin reached €120,500.75 👉 (+14.2%).",
      "driver_explanation": "Product line 'Alpha-X' (₹95.00/unit) drove +42% of volume.",
      "suggested_action": "Prioritize Alpha-X supply chain contracts."
    }
    Hope this helps!"""
    res = _parse_and_validate_insight_json(text)
    assert "€120,500.75 👉" in res["headline"]
    assert "₹95.00/unit" in res["driver_explanation"]


def test_reject_non_json_arbitrary_prose():
    prose = "Revenue is doing great. We should keep selling stuff. Talk soon!"
    with pytest.raises(OpenRouterError, match="AI response is not valid JSON"):
        _parse_and_validate_insight_json(prose)


def test_reject_missing_required_fields():
    incomplete = json.dumps({
        "headline": "Revenue is up.",
        "driver_explanation": "Sales increased.",
    })
    with pytest.raises(OpenRouterError, match="missing required field"):
        _parse_and_validate_insight_json(incomplete)


def test_reject_non_string_field_types():
    invalid_types = json.dumps({
        "headline": 12345,
        "driver_explanation": "Sales increased.",
        "suggested_action": "Keep doing it.",
    })
    with pytest.raises(OpenRouterError, match="must be strings"):
        _parse_and_validate_insight_json(invalid_types)


def test_reject_empty_or_whitespace_fields():
    empty_field = json.dumps({
        "headline": "   ",
        "driver_explanation": "Sales increased.",
        "suggested_action": "Keep doing it.",
    })
    with pytest.raises(OpenRouterError, match="empty required fields"):
        _parse_and_validate_insight_json(empty_field)


def test_reject_suspiciously_long_fields():
    long_headline = json.dumps({
        "headline": "A" * 305,
        "driver_explanation": "Sales increased.",
        "suggested_action": "Keep doing it.",
    })
    with pytest.raises(OpenRouterError, match="exceeds maximum allowed length"):
        _parse_and_validate_insight_json(long_headline)

    long_driver = json.dumps({
        "headline": "Revenue is up.",
        "driver_explanation": "B" * 605,
        "suggested_action": "Keep doing it.",
    })
    with pytest.raises(OpenRouterError, match="exceeds maximum allowed length"):
        _parse_and_validate_insight_json(long_driver)

    long_action = json.dumps({
        "headline": "Revenue is up.",
        "driver_explanation": "Sales increased.",
        "suggested_action": "C" * 405,
    })
    with pytest.raises(OpenRouterError, match="exceeds maximum allowed length"):
        _parse_and_validate_insight_json(long_action)



def test_never_cache_invalid_responses(monkeypatch, tmp_path):
    cache_file = str(tmp_path / "test_cache.json")
    monkeypatch.setattr("insight.cache.INSIGHT_CACHE_PATH", cache_file)

    def _mock_openrouter(*args, **kwargs):
        return "I am not JSON at all!", "openrouter/auto"

    monkeypatch.setattr("insight.client._call_openrouter", _mock_openrouter)
    monkeypatch.setattr("insight.budget.can_make_call", lambda: True)

    packet = {"kpi_name": "Revenue", "current_value": 100}
    with pytest.raises(OpenRouterError):
        generate_insight(packet)

    key_material = cache_key_material(packet, "openrouter/auto")
    assert insight_cache.get(key_material, path=cache_file) is None


def test_facts_builder_does_not_leak_raw_rows():
    kpi = KPIResult("Revenue", "$", 50000.0, 45000.0, 11.11, 5000.0)
    drivers = DimensionDriverResult(
        dimension="Region",
        total_delta=2000.0,
        contributions=[
            CategoryContribution("North", 3000.0, 0.0, 3000.0, 60.0),
            CategoryContribution("South", 1000.0, 2000.0, -1000.0, -20.0),
        ],
    )
    risks = [RiskSignal("spike", "Revenue spiked on 2023-01-05", "high")]

    packet = build_facts_packet(kpi, drivers, risks, "2023-01")
    assert "raw_rows" not in packet
    assert "data" not in packet
    assert packet["kpi_name"] == "Revenue"
    assert packet["current_value"] == 50000.0
    assert packet["unit"] == "$"
    assert len(packet["top_positive_drivers"]) == 1
    assert len(packet["flagged_risk_items"]) == 1


def test_reduce_facts_packet_safely_trims():
    oversized = {
        "kpi_name": "Revenue",
        "top_positive_drivers": [
            {"dimension": "Category", "category": "A" * 100, "contribution_pct": 10.0}
            for _ in range(20)
        ],
        "top_negative_drivers": [
            {"dimension": "Category", "category": "B" * 100, "contribution_pct": -10.0}
            for _ in range(20)
        ],
        "flagged_risk_items": [
            {"description": "Long risk description " * 20, "severity": "high"}
            for _ in range(20)
        ],
        "long_field": "Z" * 500,
    }
    reduced = reduce_facts_packet_if_needed(oversized, max_tokens=100)
    assert len(reduced["top_positive_drivers"]) <= 2
    assert len(reduced["top_negative_drivers"]) <= 1
    assert len(reduced["flagged_risk_items"]) <= 2
    assert len(reduced["top_positive_drivers"][0]["category"]) <= 40
    assert len(reduced["flagged_risk_items"][0]["description"]) <= 80
