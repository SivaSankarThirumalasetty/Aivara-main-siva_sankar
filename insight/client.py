"""
insight/client.py — OpenRouter API wrapper.

This is the ONLY module in the app that makes network calls. Everything to
the left of this module (ingestion, analytics, facts_builder) runs fully
offline. export/pptx_builder.py must never import this module (enforced by
convention + a unit test — see tests/test_export_no_network.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from config import (
    ALLOW_PAID_MODELS_DEFAULT,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from insight import budget, cache
from insight.facts_builder import reduce_facts_packet_if_needed
from insight.prompts import build_column_classification_prompt, build_insight_prompt, cache_key_material


@dataclass
class InsightResult:
    headline: str
    driver_explanation: str
    suggested_action: str
    raw_text: str
    from_cache: bool
    model: str


class BudgetExhaustedError(RuntimeError):
    """Raised when the daily call budget has been exhausted."""


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API call itself fails (network, auth, etc)."""


def _parse_and_validate_insight_json(raw_text: str) -> dict[str, str]:
    """Strictly parses and validates the structured JSON insight returned by LLM.

    Schema:
    {
      "headline": "...",
      "driver_explanation": "...",
      "suggested_action": "..."
    }
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise OpenRouterError("AI returned an empty response.")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    elif cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    # Extract outermost JSON object if enclosed
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    try:
        data = json.loads(cleaned)
    except Exception as exc:
        raise OpenRouterError(f"AI response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise OpenRouterError(f"AI response must be a JSON object, got {type(data).__name__}")

    # Aliases
    if "driver_explanation" not in data and "driver" in data:
        data["driver_explanation"] = data["driver"]
    if "suggested_action" not in data and "action" in data:
        data["suggested_action"] = data["action"]

    required_fields = ["headline", "driver_explanation", "suggested_action"]
    for field in required_fields:
        if field not in data:
            raise OpenRouterError(f"AI response missing required field: '{field}'")

    headline = data.get("headline")
    driver = data.get("driver_explanation")
    action = data.get("suggested_action")

    # Validate types
    if not isinstance(headline, str) or not isinstance(driver, str) or not isinstance(action, str):
        raise OpenRouterError("AI response fields must be strings.")

    headline = headline.strip()
    driver = driver.strip()
    action = action.strip()

    # Reject empty fields
    if not headline or not driver or not action:
        raise OpenRouterError("AI response contains empty required fields.")

    # Reject suspiciously long fields
    if len(headline) > 300:
        raise OpenRouterError(f"AI headline exceeds maximum allowed length ({len(headline)} > 300).")
    if len(driver) > 600:
        raise OpenRouterError(f"AI driver explanation exceeds maximum allowed length ({len(driver)} > 600).")
    if len(action) > 400:
        raise OpenRouterError(f"AI suggested action exceeds maximum allowed length ({len(action)} > 400).")

    return {
        "headline": headline,
        "driver_explanation": driver,
        "suggested_action": action,
    }


def _split_into_sentences(text: str) -> list[str]:
    try:
        parsed = _parse_and_validate_insight_json(text)
        return [parsed["headline"], parsed["driver_explanation"], parsed["suggested_action"]]
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and any(re.match(r"^(?:[1-3][\.\)]|\-|\*)\s*", l) for l in lines):
        numbered_parts = []
        for line in lines:
            cleaned_line = re.sub(r"^(?:[1-3][\.\)]|\-|\*)\s*", "", line).strip()
            if cleaned_line:
                numbered_parts.append(cleaned_line)
        if len(numbered_parts) >= 2:
            return numbered_parts

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", text.strip()) if p.strip()]
    return parts


def _call_openrouter(messages: list[dict], model: str, allow_paid: bool) -> tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    actual_model = model
    if not allow_paid and ":free" not in model and model != "openrouter/auto":
        # Guard rail: never silently call a paid model.
        actual_model = FALLBACK_MODEL

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], actual_model
    except requests.RequestException as exc:
        msg = str(exc)
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY in msg:
            msg = msg.replace(OPENROUTER_API_KEY, "[MASKED_API_KEY]")
        raise OpenRouterError(f"OpenRouter request failed: {msg}") from exc
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {exc}") from exc


def generate_insight(
    facts_packet: dict,
    model: str = DEFAULT_MODEL,
    allow_paid: bool = ALLOW_PAID_MODELS_DEFAULT,
) -> InsightResult:
    """Fail-soft insight generation:
      1. Safely compress secondary packet lists if nearing budget limits.
      2. Check token size estimate -> raise OpenRouterError if still oversized.
      3. Check cache by content hash -> return cached hit, no network call, no budget spend.
      4. Check budget -> if exhausted, raise BudgetExhaustedError.
      5. Call OpenRouter, strictly parse & validate structured JSON, cache only on success.
    """
    # 1. Reduce packet if needed
    facts_packet = reduce_facts_packet_if_needed(facts_packet, max_tokens=2500)

    # 2. Guard: estimate token size of facts packet
    packet_json = json.dumps(facts_packet, ensure_ascii=False)
    if budget.estimate_tokens(packet_json) > 3000:
        raise OpenRouterError("Facts packet exceeds safe token budget.")

    key_material = cache_key_material(facts_packet, model)
    cached = cache.get(key_material)
    if cached is not None:
        return InsightResult(**cached, from_cache=True)

    if not budget.can_make_call() or not budget.can_session_make_call():
        raise BudgetExhaustedError("Daily LLM call budget has been reached.")

    messages = build_insight_prompt(facts_packet)
    raw_text, used_model = _call_openrouter(messages, model=model, allow_paid=allow_paid)
    budget.record_call()
    budget.record_session_call()

    # Strict JSON schema validation
    parsed = _parse_and_validate_insight_json(raw_text)

    result_dict = {
        "headline": parsed["headline"],
        "driver_explanation": parsed["driver_explanation"],
        "suggested_action": parsed["suggested_action"],
        "raw_text": raw_text,
        "model": used_model,
    }
    # Cache only after strict validation succeeds
    cache.set(key_material, result_dict)

    return InsightResult(**result_dict, from_cache=False)


def classify_ambiguous_columns(
    column_payload: list[dict],
    model: str = DEFAULT_MODEL,
    allow_paid: bool = ALLOW_PAID_MODELS_DEFAULT,
) -> list[dict]:
    """Tier 2 classification fallback (Section 8.2). Budget-aware like
    insight generation; callers should treat a BudgetExhaustedError as "keep
    the Tier 1 guess" rather than a hard failure."""
    if not column_payload:
        return []

    key_material = cache_key_material({"columns": column_payload}, model)
    cached = cache.get(key_material)
    if cached is not None:
        return cached.get("classifications", [])

    if not budget.can_make_call() or not budget.can_session_make_call():
        raise BudgetExhaustedError("Daily LLM call budget has been reached.")

    messages = build_column_classification_prompt(column_payload)
    raw_text, _ = _call_openrouter(messages, model=model, allow_paid=allow_paid)
    budget.record_call()
    budget.record_session_call()

    try:
        cleaned = raw_text.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        classifications = json.loads(cleaned)
    except json.JSONDecodeError:
        classifications = []

    cache.set(key_material, {"classifications": classifications})
    return classifications
