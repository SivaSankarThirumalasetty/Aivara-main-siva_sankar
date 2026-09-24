import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd

from ingestion.loader import load_csv, load_excel, load_file

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str):
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return load_csv(f.read(), filename=name)


def test_clean_csv_loads():
    result = _load("clean.csv")
    assert result.success
    assert result.delimiter == ","
    assert list(result.dataframe.columns) == ["date", "region", "product", "revenue", "cost", "units"]
    assert len(result.dataframe) == 7


def test_semicolon_delimited_detected():
    result = _load("semicolon.csv")
    assert result.success
    assert result.delimiter == ";"
    assert "sales" in result.dataframe.columns


def test_no_header_generates_column_names():
    result = _load("no_header.csv")
    assert result.success
    assert all(c.startswith("column_") for c in result.dataframe.columns)
    assert len(result.dataframe) == 4


def test_currency_formatted_column_parses():
    result = _load("currency.csv")
    assert result.success
    assert "revenue" in result.dataframe.columns
    # revenue column is still raw strings at load time; cleaning happens in
    # ingestion/cleaner.py, so just assert the load itself didn't crash and
    # produced the right shape.
    assert len(result.dataframe) == 4


def test_mixed_encoding_does_not_crash():
    result = _load("mixed_encoding.csv")
    assert result.success
    assert len(result.dataframe) == 3


def test_no_date_column_loads_fine():
    result = _load("no_date.csv")
    assert result.success
    assert "date" not in result.dataframe.columns


def test_garbage_bytes_never_raises():
    garbage = bytes([0xFF, 0xFE, 0x00, 0x01]) * 20
    result = load_csv(garbage, filename="garbage.csv")
    # Must not raise; success may be True or False depending on how pandas
    # interprets the bytes, but it must always return a LoadResult.
    assert result is not None


def test_load_excel_single_sheet():
    df_in = pd.DataFrame({"product": ["A", "B", "C"], "revenue": [100, 200, 300]})
    buf = io.BytesIO()
    df_in.to_excel(buf, index=False, engine="openpyxl")
    excel_bytes = buf.getvalue()

    result = load_excel(excel_bytes, filename="sample.xlsx")
    assert result.success
    assert result.delimiter == "excel"
    assert list(result.dataframe.columns) == ["product", "revenue"]
    assert len(result.dataframe) == 3


def test_load_excel_multi_sheet():
    df1 = pd.DataFrame({"month": ["Jan", "Feb"], "sales": [10, 20]})
    df2 = pd.DataFrame({"month": ["Mar", "Apr"], "sales": [30, 40]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Q1", index=False)
        df2.to_excel(writer, sheet_name="Q2", index=False)
    excel_bytes = buf.getvalue()

    result = load_excel(excel_bytes, filename="quarterly.xlsx")
    assert result.success
    assert any("Workbook has 2 sheets" in note for note in result.notes)
    assert list(result.dataframe["sales"]) == [10, 20]


def test_load_file_routes_xlsx_and_csv():
    # Test CSV routing
    csv_bytes = b"a,b\n1,2\n"
    res_csv = load_file(csv_bytes, filename="test.csv")
    assert res_csv.success
    assert res_csv.delimiter == ","

    # Test XLSX routing
    buf = io.BytesIO()
    pd.DataFrame({"x": [1, 2]}).to_excel(buf, index=False, engine="openpyxl")
    res_xlsx = load_file(buf.getvalue(), filename="test.xlsx")
    assert res_xlsx.success
    assert res_xlsx.delimiter == "excel"
    assert "x" in res_xlsx.dataframe.columns

def test_malformed_csv_rows_reported_in_notes():
    # CSV with clear header and numeric data rows, with 1 malformed row in the middle
    csv_bytes = b"product,revenue\nWidget A,100.0\nWidget B,200.0\nBadRow,123.0,ExtraCol1,ExtraCol2\nWidget C,300.0\nWidget D,400.0\n"
    res = load_csv(csv_bytes, filename="malformed.csv")
    assert res.success
    assert len(res.dataframe) == 4
    assert any("skipped: malformed" in note for note in res.notes)
