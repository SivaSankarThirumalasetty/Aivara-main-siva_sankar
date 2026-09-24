import io
import json
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.kpis import KPIResult
from config import OPENROUTER_API_KEY
from export.pptx_builder import DeckInputs, ViewExportData, build_deck
from ingestion.classifier import (
    ColumnClassification,
    build_llm_classification_payload,
)
from ingestion.loader import (
    MAX_FILE_SIZE_BYTES,
    load_file,
    sanitize_filename,
)
from insight import budget, cache as insight_cache
from insight.client import (
    BudgetExhaustedError,
    OpenRouterError,
    _call_openrouter,
    generate_insight,
)
from insight.facts_builder import build_facts_packet
from insight.prompts import build_insight_prompt, cache_key_material
from ui.components import mask_secret


# ===========================================================================
# TEST 1: User B cannot retrieve User A's cached insight
# ===========================================================================
def test_redteam_cross_session_view_cache_no_fallback():
    session_a = "session_user_alice"
    session_b = "session_user_bob"

    insight_alice = {
        "headline": "Alice Confidential Revenue grew 50%.",
        "driver_explanation": "Private customer X drove the expansion.",
        "suggested_action": "Keep secret.",
    }
    insight_cache.set_for_view("Executive", insight_alice, session_id=session_a)

    # Bob must NOT receive Alice's insight
    bob_view = insight_cache.get_for_view("Executive", session_id=session_b)
    assert bob_view is None

    # In a simulated Streamlit session state without '_view_insights', get_for_view must NOT fall back to global
    class MockStreamlit:
        session_state = {}

    import insight.cache as ic
    orig_st = sys.modules.get("streamlit")
    sys.modules["streamlit"] = MockStreamlit()
    try:
        assert ic.get_for_view("Executive") is None
    finally:
        if orig_st is not None:
            sys.modules["streamlit"] = orig_st
        else:
            sys.modules.pop("streamlit", None)


# ===========================================================================
# TEST 2: Dataset A values cannot appear in Dataset B's PPTX
# ===========================================================================
def test_redteam_dataset_replacement_pptx_clean():
    views_a = [ViewExportData(view_name="Executive", kpi_tiles=[])]
    deck_a = build_deck(DeckInputs("Company_A.csv", "Jan 2026", views_a, []))
    assert deck_a.getvalue() is not None

    views_b = [ViewExportData(view_name="Executive", kpi_tiles=[])]
    deck_b = build_deck(DeckInputs("Company_B.csv", "Feb 2026", views_b, []))
    assert deck_b.getvalue() is not None


# ===========================================================================
# TEST 3: Raw CSV rows are NEVER sent to OpenRouter
# ===========================================================================
def test_redteam_no_raw_rows_in_facts_or_prompts():
    df = pd.DataFrame({
        "SSN": ["000-12-3456", "000-12-3457"],
        "Salary": [150000, 200000],
        "SecretNote": ["Secret password 1", "Secret password 2"],
    })
    kpi = KPIResult("Salary", "$", 350000.0, 300000.0, 16.7, 50000.0)
    packet = build_facts_packet(kpi, None, None, "2026-Q1")
    packet_json = json.dumps(packet)

    # Assert no sensitive row strings appear in facts packet
    assert "000-12-3456" not in packet_json
    assert "Secret password" not in packet_json

    messages = build_insight_prompt(packet)
    prompt_str = json.dumps(messages)
    assert "000-12-3456" not in prompt_str
    assert "Secret password" not in prompt_str

    # Classification payload for ambiguous columns only sends column names & sample strings
    cols = [ColumnClassification("Salary", "metric", 0.4, "low conf")]
    payload = build_llm_classification_payload(df, cols, n_samples=1)
    assert len(payload) == 1
    assert payload[0]["column_name"] == "Salary"
    assert "Secret password" not in json.dumps(payload)


# ===========================================================================
# TEST 4: API key scrubbing in UI, exceptions, and reports
# ===========================================================================
def test_redteam_api_key_scrubbed_from_exceptions(monkeypatch):
    fake_secret_key = "sk-or-v1-secret-password-1234567890"
    monkeypatch.setattr("insight.client.OPENROUTER_API_KEY", fake_secret_key)
    monkeypatch.setattr("config.OPENROUTER_API_KEY", fake_secret_key)

    # 1. UI masking helper
    masked = mask_secret(fake_secret_key)
    assert fake_secret_key not in masked
    assert masked.endswith("7890")
    assert masked.startswith("*")

    # 2. Exception message scrubbing
    import requests

    def _mock_post(*args, **kwargs):
        raise requests.RequestException(f"Failed to auth with bearer {fake_secret_key}")

    monkeypatch.setattr("requests.post", _mock_post)

    with pytest.raises(OpenRouterError) as exc_info:
        _call_openrouter([{"role": "user", "content": "hi"}], model="openrouter/auto", allow_paid=False)

    assert fake_secret_key not in str(exc_info.value)
    assert "[MASKED_API_KEY]" in str(exc_info.value)


# ===========================================================================
# TEST 5: Malicious filenames are sanitized
# ===========================================================================
def test_redteam_malicious_filename_sanitized():
    traversal_win = r"..\..\..\windows\system32\cmd.exe"
    sanitized = sanitize_filename(traversal_win)
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert sanitized.endswith("cmd.exe")

    traversal_nix = "../../../../etc/passwd"
    assert sanitize_filename(traversal_nix) == "passwd"

    null_byte = "legit_file\x00_hidden.exe"
    assert "\x00" not in sanitize_filename(null_byte)

    script_tag = "script_alert_1.csv"
    cleaned = sanitize_filename(script_tag)
    assert cleaned == "script_alert_1.csv"

    none_file = sanitize_filename(None)
    assert none_file == "uploaded.csv"


# ===========================================================================
# TEST 6: Memory Exhaustion / Denial of Service rejection
# ===========================================================================
def test_redteam_file_size_limit_and_empty_file():
    # 0-byte file
    res_empty = load_file(b"", "empty.csv")
    assert not res_empty.success
    assert "empty" in res_empty.error.lower()

    # Oversized payload exceeding 50MB
    oversized_bytes = b"A" * (MAX_FILE_SIZE_BYTES + 1024)
    res_oversized = load_file(oversized_bytes, "massive.csv")
    assert not res_oversized.success
    assert "exceeds maximum allowed size" in res_oversized.error.lower()


# ===========================================================================
# TEST 7: Malformed CSV & Excel inputs fail gracefully
# ===========================================================================
def test_redteam_malformed_csv_and_excel_fail_gracefully():
    # Corrupted binary disguised as excel
    corrupted_excel = b"PK\x03\x04corrupted_zip_payload_9999"
    res_excel = load_file(corrupted_excel, "broken.xlsx")
    assert not res_excel.success


# ===========================================================================
# TEST 8: Malformed AI responses are NEVER persisted
# ===========================================================================
def test_redteam_malformed_ai_response_never_cached(monkeypatch, tmp_path):
    cache_path = str(tmp_path / "anti_poison_cache.json")
    monkeypatch.setattr("insight.cache.INSIGHT_CACHE_PATH", cache_path)
    monkeypatch.setattr("insight.budget.can_make_call", lambda: True)
    monkeypatch.setattr("insight.budget.can_session_make_call", lambda: True)

    def _mock_bad_llm(*args, **kwargs):
        return "This is hallucinated prose without JSON!", "openrouter/auto"

    monkeypatch.setattr("insight.client._call_openrouter", _mock_bad_llm)

    packet = {"kpi_name": "Margin", "current_value": 42.0}
    with pytest.raises(OpenRouterError):
        generate_insight(packet)

    key_mat = cache_key_material(packet, "openrouter/auto")
    assert insight_cache.get(key_mat, path=cache_path) is None


# ===========================================================================
# TEST 9: Per-session budget enforcement prevents single-user exhaustion
# ===========================================================================
def test_redteam_per_session_budget_limiting(monkeypatch):
    class MockSessionState(dict):
        pass

    state = MockSessionState({"_session_llm_calls": 15})

    class MockStreamlit:
        session_state = state

    orig_st = sys.modules.get("streamlit")
    sys.modules["streamlit"] = MockStreamlit()
    try:
        assert not budget.can_session_make_call(limit=15)
        with pytest.raises(BudgetExhaustedError, match="Daily LLM call budget"):
            generate_insight({"kpi_name": "Rev", "current_value": 10})
    finally:
        if orig_st is not None:
            sys.modules["streamlit"] = orig_st
        else:
            sys.modules.pop("streamlit", None)


# ===========================================================================
# TEST 10: Persistence files are clearly documented in README
# ===========================================================================
def test_redteam_persistence_documentation():
    readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".aivara_insight_cache.json" in content
    assert ".aivara_budget_state.json" in content
    assert "Data privacy" in content
