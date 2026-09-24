"""
ingestion/cleaner.py — null handling, dedupe, whitespace/type coercion.

Cleaning never silently drops or alters rows: every step that changes the
data returns a count so the UI can tell the user exactly what happened
(Section 12: "No silent data loss").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_CURRENCY_CHARS = re.compile(r"[₹$€£,]")
_PERCENT_CHAR = re.compile(r"%\s*$")
_PAREN_NEGATIVE = re.compile(r"^\((.*)\)$")
_DATE_SHAPE_PATTERN = re.compile(
    r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(\s|$|T)|^\s*\d{4}\d{2}\d{2}\s*$"
)
_NULL_STRINGS = {"", "na", "n/a", "nan", "none", "null", "-", "--"}


@dataclass
class CleanReport:
    empty_rows_removed: int = 0
    empty_columns_removed: int = 0
    duplicate_rows_flagged: int = 0
    columns_coerced_numeric: list[str] = field(default_factory=list)
    columns_coerced_datetime: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _try_numeric_coerce(series: pd.Series) -> tuple[pd.Series, bool, int]:
    """Attempt to coerce a string-ish column that looks numeric (currency
    symbols, thousands separators, parens-as-negative, trailing %) into a
    proper numeric dtype. Returns (converted_series, did_convert, unparsed_count)."""
    if pd.api.types.is_numeric_dtype(series):
        return series, False, 0

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return series, False, 0

    # Guard: do not coerce if the column matches date formats (e.g. YYYYMMDD)
    date_matches = non_null.map(lambda v: bool(_DATE_SHAPE_PATTERN.search(v)))
    if date_matches.mean() >= 0.8:
        return series, False, 0

    def clean_value(v: str) -> str:
        v = v.strip()
        is_percent = bool(_PERCENT_CHAR.search(v))
        v = _PERCENT_CHAR.sub("", v)
        paren = _PAREN_NEGATIVE.match(v)
        negative = paren is not None
        if paren:
            v = paren.group(1)
        v = _CURRENCY_CHARS.sub("", v).strip()
        if negative and v and not v.startswith("-"):
            v = "-" + v
        if is_percent and v:
            try:
                v = str(float(v) / 100.0)
            except ValueError:
                pass
        return v

    cleaned = non_null.map(clean_value)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_frac = numeric.notna().mean() if len(numeric) else 0
    if success_frac >= 0.9:
        full_cleaned = series.map(lambda v: clean_value(str(v)) if pd.notna(v) else v)
        converted = pd.to_numeric(full_cleaned, errors="coerce")
        unparsed_count = int(series.notna().sum() - converted.notna().sum())
        return converted, True, unparsed_count
    return series, False, 0


def _stringy_columns(df: pd.DataFrame) -> list[str]:
    """Object-dtype OR pandas' dedicated string dtypes — dtype naming for
    text columns varies across pandas versions (pandas 2's nullable
    'string' dtype, pandas 3's default 'str' dtype), so check semantically
    rather than comparing to `object` alone."""
    return [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    report = CleanReport()
    df = df.copy()

    # 1. Trim whitespace on string/object columns and normalize null tokens to NaN.
    def _normalize_null_value(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            v_clean = v.strip()
            if v_clean.lower() in _NULL_STRINGS:
                return np.nan
            return v_clean
        return v

    for col in _stringy_columns(df):
        df[col] = df[col].apply(_normalize_null_value)

    # 2. Drop fully-empty or uninformative spacer columns.
    empty_cols = [
        c for c in df.columns
        if df[c].isna().all()
        or (str(c).startswith("Unnamed:") and (df[c].isna().mean() > 0.85 or df[c].dropna().nunique() <= 1))
        or (df[c].isna().mean() > 0.95 and df[c].dropna().nunique() <= 1 and c != "_is_duplicate")
    ]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        report.empty_columns_removed = len(empty_cols)
        report.messages.append(f"{len(empty_cols)} empty/uninformative spacer column(s) removed.")

    # 3. Drop fully-empty rows.
    before = len(df)
    df = df.dropna(how="all")
    removed = before - len(df)
    if removed:
        report.empty_rows_removed = removed
        report.messages.append(f"{removed} fully-empty row(s) removed.")

    # 4. Flag (do not drop) duplicate rows.
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count:
        report.duplicate_rows_flagged = dup_count
        report.messages.append(
            f"{dup_count} duplicate row(s) flagged (kept in data, marked via '_is_duplicate')."
        )
        df["_is_duplicate"] = dup_mask
    else:
        df["_is_duplicate"] = False

    # 5. Attempt numeric coercion on object columns that look like formatted numbers.
    stringy_now = set(_stringy_columns(df))
    numeric_details = []
    for col in df.columns:
        if col == "_is_duplicate" or col not in stringy_now:
            continue
        converted, did, unparsed_count = _try_numeric_coerce(df[col])
        if did:
            df[col] = converted
            report.columns_coerced_numeric.append(col)
            if unparsed_count > 0:
                numeric_details.append(
                    f"{col} ({unparsed_count} value(s) could not be parsed and are now empty)"
                )
            else:
                numeric_details.append(col)

    if report.columns_coerced_numeric:
        report.messages.append(
            "Coerced to numeric (currency/percent/comma formatting stripped): "
            + ", ".join(numeric_details)
        )

    # 6. Attempt datetime coercion on remaining object columns with plausible names/values.
    stringy_now = set(_stringy_columns(df))
    datetime_details = []
    for col in df.columns:
        if col == "_is_duplicate" or col not in stringy_now:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        success_frac = parsed.notna().mean()
        if success_frac >= 0.8:
            unparsed_count = int(non_null.shape[0] - parsed.notna().sum())
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            report.columns_coerced_datetime.append(col)
            if unparsed_count > 0:
                datetime_details.append(
                    f"{col} ({unparsed_count} value(s) could not be parsed and are now empty)"
                )
            else:
                datetime_details.append(col)

    if report.columns_coerced_datetime:
        report.messages.append(
            "Parsed as dates: " + ", ".join(datetime_details)
        )

    return df, report
