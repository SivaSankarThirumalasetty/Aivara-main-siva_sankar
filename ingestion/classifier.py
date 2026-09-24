"""
ingestion/classifier.py — semantic column role classification.

Tier 1 (this module, always runs, no network): rule-based classification into
one of: date, metric, dimension, id, text/unclassified.

Tier 2 (insight/client.py, budget-aware): LLM fallback for genuinely
ambiguous columns only — see classify_ambiguous() at the bottom, which
prepares the small "column name + sample values" payload but does not itself
call the network (that happens in insight/client.py so all network calls live
in one module).

Every classification carries a confidence score so the UI can flag low-
confidence columns for the LLM fallback / manual override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from config import (
    DATE_PARSE_SUCCESS_THRESHOLD,
    DIMENSION_UNIQUE_ABS_CAP,
    DIMENSION_UNIQUE_RATIO_CAP,
    ID_UNIQUE_RATIO_FLOOR,
)

DATE_NAME_PATTERN = re.compile(r"(date|_dt$|^dt_|period|month|year|timestamp|time)", re.I)
ID_NAME_PATTERN = re.compile(r"(^id$|[_\s]id$|^uuid$|[_\s]uuid$|[_\s]code$|^code$|[_\s]sku$|^sku$|[_\s]num$|^num$|#$)", re.I)
LABEL_NAME_PATTERN = re.compile(
    r"(name|title|desc|description|label|category|type|assigned|owner|person|employee|segment|status|state|team)",
    re.I,
)
RATE_NAME_PATTERN = re.compile(
    r"(rate|pct|percent|ratio|avg|average|score|margin|progress|completion|prob|probability|discount)",
    re.I,
)

ROLE_DATE = "date"
ROLE_METRIC = "metric"
ROLE_DIMENSION = "dimension"
ROLE_ID = "id"
ROLE_TEXT = "text"

LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class ColumnClassification:
    name: str
    role: str
    confidence: float
    reason: str
    is_rate: bool = False


def _is_stringy(series: pd.Series) -> bool:
    """True for classic object-dtype string columns AND pandas' newer
    dedicated string dtypes (pandas >= 2's 'string' dtype, and pandas 3's
    default 'str' dtype) — dtype naming for text columns varies by pandas
    version, so we check semantically rather than comparing to `object`."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _classify_column(name: str, series: pd.Series, n_rows: int) -> ColumnClassification:
    non_null = series.dropna()
    n_unique = non_null.nunique()
    unique_ratio = (n_unique / n_rows) if n_rows else 0.0
    name_matches_date = bool(DATE_NAME_PATTERN.search(name))
    name_matches_id = bool(ID_NAME_PATTERN.search(name))
    name_matches_label = bool(LABEL_NAME_PATTERN.search(name))
    stringy = _is_stringy(series)
    numeric = pd.api.types.is_numeric_dtype(series)

    # --- date check --------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnClassification(name, ROLE_DATE, 0.99, "Already a datetime dtype.")

    if stringy or name_matches_date:
        sample = non_null.astype(str) if stringy else non_null
        if not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            success = parsed.notna().mean()
            if success >= DATE_PARSE_SUCCESS_THRESHOLD:
                conf = min(0.95, 0.6 + success * 0.35)
                return ColumnClassification(
                    name, ROLE_DATE, conf, f"{success:.0%} of values parse as dates."
                )
            if name_matches_date and success >= 0.4:
                return ColumnClassification(
                    name,
                    ROLE_DATE,
                    0.5,
                    f"Name suggests a date column but only {success:.0%} parse cleanly.",
                )

    # --- numeric columns: metric, coded dimension, or (rarely) numeric id ---
    if numeric:
        is_rate = bool(RATE_NAME_PATTERN.search(name))
        if not non_null.empty:
            all_in_unit_range = (non_null >= 0.0).all() and (non_null <= 1.0).all()
            has_decimals = ((non_null % 1) != 0).any()
            if all_in_unit_range and has_decimals:
                is_rate = True

        if name_matches_id and not is_rate and unique_ratio >= 0.5 and n_rows > 20:
            return ColumnClassification(
                name, ROLE_ID, 0.7, "Numeric but column name matches id/code pattern."
            )

        low_cardinality_code = n_unique <= max(20, int(0.02 * n_rows)) and n_unique < 15
        if low_cardinality_code and n_rows > 50 and not is_rate:
            return ColumnClassification(
                name,
                ROLE_DIMENSION,
                0.55,
                f"Numeric but low-cardinality ({n_unique} distinct values) — likely a coded category.",
            )

        conf = 0.95 if is_rate else 0.9
        reason = "Percentage/rate metric." if is_rate else "Numeric dtype with continuous-looking variation."
        return ColumnClassification(name, ROLE_METRIC, conf, reason, is_rate=is_rate)

    # --- string columns: id, dimension, or text -----------------------------
    if stringy:
        # Check explicit ID match first (e.g. 'product_id', 'customer_id', 'sku')
        if name_matches_id and unique_ratio >= 0.5:
            return ColumnClassification(
                name, ROLE_ID, 0.9, "Column name matches id/code pattern with high uniqueness."
            )

        # If name indicates a human-facing label (e.g. 'product_name', 'task_name', 'assigned_to')
        # or dataset has fewer than 150 rows (small catalog, team list, etc.), it's a dimension!
        if name_matches_label:
            return ColumnClassification(
                name,
                ROLE_DIMENSION,
                0.9,
                f"Entity label/category name with {n_unique} distinct values.",
            )

        if unique_ratio >= ID_UNIQUE_RATIO_FLOOR and n_rows > 100:
            return ColumnClassification(
                name, ROLE_ID, 0.75, f"Near-unique per row ({unique_ratio:.0%} unique) in large dataset."
            )

        is_low_cardinality = (
            unique_ratio < DIMENSION_UNIQUE_RATIO_CAP or n_unique <= DIMENSION_UNIQUE_ABS_CAP or n_rows <= 150
        )
        if is_low_cardinality and n_unique > 0:
            conf = 0.85 if unique_ratio < DIMENSION_UNIQUE_RATIO_CAP else 0.6
            return ColumnClassification(
                name, ROLE_DIMENSION, conf, f"{n_unique} distinct values ({unique_ratio:.1%} unique)."
            )

    # --- fallback ----------------------------------------------------------
    return ColumnClassification(
        name,
        ROLE_TEXT,
        0.4,
        "Did not clearly match date/metric/dimension/id heuristics.",
    )


def classify_columns(df: pd.DataFrame) -> list[ColumnClassification]:
    """Rule-based (Tier 1) classification for every column in df."""
    n_rows = len(df)
    results = []
    for col in df.columns:
        if col == "_is_duplicate":
            continue
        results.append(_classify_column(col, df[col], n_rows))
    return results


def ambiguous_columns(
    classifications: list[ColumnClassification], threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> list[ColumnClassification]:
    """Columns worth sending to the Tier 2 LLM fallback."""
    return [c for c in classifications if c.confidence < threshold]


def build_llm_classification_payload(
    df: pd.DataFrame, columns: list[ColumnClassification], n_samples: int = 5
) -> list[dict]:
    """Prepares the minimal payload for Tier 2 classification: column name +
    a handful of sample values only — never the full column, never other
    columns. The actual network call happens in insight/client.py."""
    payload = []
    for c in columns:
        sample_values = df[c.name].dropna().astype(str).head(n_samples).tolist()
        payload.append(
            {
                "column_name": c.name,
                "sample_values": sample_values,
                "tier1_guess": c.role,
                "tier1_confidence": round(c.confidence, 2),
            }
        )
    return payload


def apply_overrides(
    classifications: list[ColumnClassification], overrides: dict[str, str]
) -> list[ColumnClassification]:
    """Apply user-chosen overrides from the UI on top of the classifier
    output. `overrides` maps column name -> role string."""
    updated = []
    for c in classifications:
        if c.name in overrides and overrides[c.name] != c.role:
            updated.append(
                ColumnClassification(c.name, overrides[c.name], 1.0, "Manually overridden by user.")
            )
        else:
            updated.append(c)
    return updated
