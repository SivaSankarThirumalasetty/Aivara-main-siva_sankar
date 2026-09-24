import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ingestion.cleaner import clean_dataframe


def test_percent_columns_scaled_properly():
    df = pd.DataFrame({"growth": ["45%", "50%", "12%"]})
    cleaned_df, report = clean_dataframe(df)
    assert pd.api.types.is_numeric_dtype(cleaned_df["growth"])
    # 45% should be 0.45, not 45
    assert abs(cleaned_df["growth"].iloc[0] - 0.45) < 1e-6
    assert abs(cleaned_df["growth"].iloc[1] - 0.50) < 1e-6
    assert abs(cleaned_df["growth"].iloc[2] - 0.12) < 1e-6


def test_null_placeholders_normalized():
    tokens = ["", "na", "n/a", "N/A", "NaN", "NAN", "None", "none", "NULL", "null", "-", "--", "Valid"]
    df = pd.DataFrame({"category": tokens, "val": range(len(tokens))})
    cleaned_df, report = clean_dataframe(df)
    assert "category" in cleaned_df
    assert cleaned_df["category"].iloc[:-1].isna().all()
    assert cleaned_df["category"].iloc[-1] == "Valid"


def test_partial_coercion_reports_failure_count():
    # 9 numbers, 1 stray text -> 90% clean, should coerce and report 1 unparsed value
    vals = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "corrupted"]
    df = pd.DataFrame({"revenue": vals})
    cleaned_df, report = clean_dataframe(df)
    assert pd.api.types.is_numeric_dtype(cleaned_df["revenue"])
    assert any("could not be parsed and are now empty" in m for m in report.messages)


def test_compact_date_format_coerced_to_datetime_not_numeric():
    dates = ["20240101", "20240102", "20240103", "20240104"]
    df = pd.DataFrame({"date_col": dates, "revenue": ["100", "200", "300", "400"]})
    cleaned_df, report = clean_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date_col"])
    assert pd.api.types.is_numeric_dtype(cleaned_df["revenue"])
