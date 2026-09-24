"""
ingestion/loader.py — robust CSV loading.

Handles, without crashing and without asking the user to pre-clean the file:
  * unknown delimiter (comma, semicolon, tab, pipe)
  * unknown encoding (utf-8, utf-8-sig, detected, latin-1 fallback)
  * uncertain header row (first row may not be a header)

Returns a LoadResult with the DataFrame plus a human-readable log of what
was detected/assumed, so the UI can show the user what happened.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from config import MAX_ROWS_FULL_PROCESSING

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB hard safety limit


def sanitize_filename(filename: str | None) -> str:
    """Sanitizes user-supplied filenames to prevent path traversal, control character injection,
    and presentation formatting issues."""
    if not filename or not isinstance(filename, str):
        return "uploaded.csv"

    # Extract base filename to prevent path traversal
    base = os.path.basename(filename.replace("\\", "/"))

    # Strip control chars, null bytes, and non-printable characters
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", base)

    # Allow alphanumeric, spaces, dots, dashes, underscores, and common punctuation
    cleaned = re.sub(r"[^\w\s\.\-_]", "_", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned or cleaned == ".":
        return "uploaded.csv"

    # Bound maximum length
    if len(cleaned) > 100:
        name_part, ext = os.path.splitext(cleaned)
        cleaned = name_part[: 100 - len(ext)] + ext

    return cleaned

try:
    from charset_normalizer import from_bytes as _cn_from_bytes
except ImportError:  # pragma: no cover - optional dependency
    _cn_from_bytes = None


CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig"]


@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    encoding: str
    delimiter: str
    header_row: int | None
    notes: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


def _detect_encoding(raw: bytes) -> tuple[str, list[str]]:
    """Try a fallback chain of encodings and return the first that decodes cleanly."""
    notes: list[str] = []
    for enc in CANDIDATE_ENCODINGS:
        try:
            raw.decode(enc)
            return enc, notes
        except (UnicodeDecodeError, LookupError):
            continue

    if _cn_from_bytes is not None:
        try:
            best = _cn_from_bytes(raw).best()
            if best is not None:
                enc = best.encoding
                raw.decode(enc)
                notes.append(f"Encoding auto-detected as {enc}.")
                return enc, notes
        except Exception:  # pragma: no cover - defensive
            pass

    # Last resort: latin-1 never raises (every byte value is a valid code point).
    notes.append("Could not confidently detect encoding; falling back to latin-1.")
    return "latin-1", notes


def _detect_delimiter(sample_text: str) -> tuple[str, list[str]]:
    """Use csv.Sniffer first; fall back to picking the delimiter with the most
    consistent column count across sample lines."""
    notes: list[str] = []
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(CANDIDATE_DELIMITERS))
        return dialect.delimiter, notes
    except csv.Error:
        notes.append("csv.Sniffer could not confidently detect a delimiter; trying candidates.")

    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:50]
    if not lines:
        return ",", notes + ["No sample lines available; defaulting to comma."]

    best_delim = ","
    best_score = -1.0
    for delim in CANDIDATE_DELIMITERS:
        counts = [ln.count(delim) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        # Score = how consistent counts are across lines (lower variance is
        # better) combined with having a non-trivial number of fields.
        mean = sum(counts) / len(counts)
        if mean == 0:
            continue
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        score = mean - variance
        if score > best_score:
            best_score = score
            best_delim = delim

    notes.append(f"Delimiter auto-detected as {best_delim!r}.")
    return best_delim, notes


_DATE_SHAPE_PATTERN = re.compile(
    r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(\s|$|T)|^\s*\d{4}\d{2}\d{2}\s*$"
)


def _is_numeric_value(val: object) -> bool:
    if isinstance(val, (int, float, complex)):
        return not pd.isna(val)
    val_str = str(val).strip().replace(",", "")
    if not val_str:
        return False
    try:
        float(val_str)
        return True
    except ValueError:
        return False


def _value_shape(val: object) -> str:
    """Classifies a single cell as 'numeric', 'date', or 'text' — used to
    compare the candidate header row against the data rows column-by-column,
    since an aggregate numeric-fraction comparison is fooled whenever a
    data-only file happens to mix text and numeric columns in a way that
    matches the row-1-vs-rest ratio (e.g. date, category, number, number)."""
    if pd.isna(val):
        return "text"
    if isinstance(val, (pd.Timestamp, pd.Period)) or hasattr(val, "strftime"):
        return "date"
    if isinstance(val, (int, float)):
        return "numeric"
    val_str = str(val).strip()
    if _is_numeric_value(val_str):
        return "numeric"
    if _DATE_SHAPE_PATTERN.match(val_str):
        return "date"
    return "text"


def _looks_like_header(candidate_row: list[str], data_rows: list[list[str]]) -> bool:
    """Heuristic: compare the candidate first row against the data rows on a
    per-column basis. Header rows are text labels; when a column's data
    values are clearly numeric or date-shaped but the candidate row's value
    in that same column is also numeric/date-shaped in the same way, the
    candidate row is itself data, not a header. Text-shaped columns (e.g. a
    category column) are uninformative for this check and are skipped, since
    a header label and a category value are both just "text"."""
    if not candidate_row:
        return True
    if not data_rows:
        return True  # can't compare against anything; default to assuming a header

    candidate_shapes = [_value_shape(v) for v in candidate_row]
    sample = data_rows[: min(10, len(data_rows))]

    matches = 0
    compared = 0
    for i, candidate_shape in enumerate(candidate_shapes):
        column_values = [row[i] for row in sample if i < len(row)]
        if not column_values:
            continue
        column_shapes = [_value_shape(v) for v in column_values]
        majority_shape = max(set(column_shapes), key=column_shapes.count)
        if majority_shape not in ("numeric", "date"):
            continue  # text columns don't help distinguish header vs. data
        compared += 1
        if candidate_shape == majority_shape:
            matches += 1

    if compared == 0:
        return True  # no numeric/date columns to compare against; assume header

    match_frac = matches / compared
    # If most of the informative (numeric/date) columns show the candidate
    # row matching the data rows' shape, the candidate row is data, not a
    # header.
    return match_frac < 0.5


def _find_best_header_row(rows: list[list[str]], max_scan: int = 25) -> tuple[int, bool]:
    """Find the best header row index in a list of candidate raw rows.
    Returns (header_row_index, has_header_boolean).
    If rows appear to be pure data with no real header, returns (0, False)."""
    if not rows:
        return 0, True

    best_row = 0
    best_score = -1.0
    scan_limit = min(max_scan, len(rows))

    for idx in range(scan_limit):
        row = rows[idx]
        non_nulls = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() != ""]
        if not non_nulls:
            continue
        text_cells = [
            v for v in non_nulls
            if not _is_numeric_value(v) and not _DATE_SHAPE_PATTERN.match(v)
        ]
        distinct_text = set(text_cells)
        # Score header based on count of valid column labels and variety
        score = len(non_nulls) * 1.5 + len(distinct_text) * 2.0
        if len(non_nulls) >= 2 and len(text_cells) == len(non_nulls):
            score += 10.0
        # If subsequent row has numbers or dates, strong signal that current row is a header
        if idx + 1 < len(rows):
            next_row = rows[idx + 1]
            next_non_nulls = [str(v).strip() for v in next_row if pd.notna(v) and str(v).strip() != ""]
            next_nums = [
                v for v in next_non_nulls
                if _is_numeric_value(v) or _DATE_SHAPE_PATTERN.match(v)
            ]
            if next_nums and len(text_cells) >= len(non_nulls) * 0.5:
                score += 8.0

        if score > best_score:
            best_score = score
            best_row = idx

    is_header = _looks_like_header(rows[best_row], rows[best_row + 1 : best_row + 11])
    return best_row, is_header


def _sanitize_dataframe_columns_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans up DataFrame after loading:
    1. Sanitizes column headers (strips internal newlines, multi-spaces, trims).
    2. Drops fully-empty rows and columns.
    3. Drops unnamed spacer columns that have >85% nulls.
    """
    if df.empty:
        return df

    # Drop completely empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Sanitize column names: remove newlines, carriage returns, collapse spaces
    cleaned_cols = []
    for c in df.columns:
        c_str = re.sub(r"[\r\n\t]+", " ", str(c))
        c_str = re.sub(r"\s+", " ", c_str).strip()
        cleaned_cols.append(c_str)
    df.columns = cleaned_cols

    # Drop unnamed columns that are mostly NaN
    cols_to_drop = [
        c for c in df.columns
        if str(c).startswith("Unnamed:") and (df[c].isna().mean() > 0.85 or df[c].dropna().nunique() <= 1)
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def load_csv(file_bytes: bytes, filename: str = "uploaded.csv") -> LoadResult:
    """Best-effort robust CSV load. Never raises; failures are reported via
    LoadResult.success / .error so the UI can show a friendly message."""
    notes: list[str] = []

    encoding, enc_notes = _detect_encoding(file_bytes)
    notes.extend(enc_notes)

    try:
        text = file_bytes.decode(encoding, errors="replace")
    except Exception as exc:  # pragma: no cover - extremely defensive
        return LoadResult(
            dataframe=pd.DataFrame(),
            encoding=encoding,
            delimiter=",",
            header_row=None,
            notes=notes,
            success=False,
            error=f"Could not decode file even with fallback handling: {exc}",
        )

    sample = "\n".join(text.splitlines()[:200])
    delimiter, delim_notes = _detect_delimiter(sample)
    notes.extend(delim_notes)

    # Peek at the first rows to decide whether a header exists and which row it is.
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i > 25:
                break
    except Exception:
        rows = []

    best_header_idx, has_header = _find_best_header_row(rows)
    if not has_header:
        notes.append("First row looks like data, not a header — generated column names instead.")
        header_arg: int | None = None
    else:
        header_arg = best_header_idx
        if best_header_idx > 0:
            notes.append(f"Auto-detected table header at row {best_header_idx + 1}.")

    skipped_bad_lines = 0

    def bad_line_handler(bad_line: list[str]) -> None:
        nonlocal skipped_bad_lines
        skipped_bad_lines += 1
        return None

    read_attempts = [
        dict(sep=delimiter, header=header_arg, engine="python"),
        dict(sep=delimiter, header=header_arg, engine="python", on_bad_lines=bad_line_handler),
        dict(sep=None, header=header_arg, engine="python", on_bad_lines=bad_line_handler),
    ]

    last_error: Exception | None = None
    for attempt_kwargs in read_attempts:
        skipped_bad_lines = 0
        try:
            df = pd.read_csv(io.StringIO(text), **attempt_kwargs)
            if df.shape[1] == 1 and delimiter != "," and attempt_kwargs.get("sep") == delimiter:
                # Only one column parsed out — probably picked the wrong
                # delimiter after all; let the next attempt (sep=None, let
                # pandas sniff) have a chance.
                last_error = ValueError("Only one column parsed; delimiter guess may be wrong.")
                continue
            if not has_header:
                df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
            if skipped_bad_lines > 0:
                notes.append(f"{skipped_bad_lines} row(s) skipped: malformed (wrong column count).")

            df = _sanitize_dataframe_columns_and_cleanup(df)

            if len(df) > MAX_ROWS_FULL_PROCESSING:
                notes.append(
                    f"Dataset exceeds {MAX_ROWS_FULL_PROCESSING:,} rows; sampled down for responsiveness."
                )
                df = df.sample(n=MAX_ROWS_FULL_PROCESSING, random_state=42).sort_index()
            return LoadResult(
                dataframe=df,
                encoding=encoding,
                delimiter=delimiter,
                header_row=header_arg,
                notes=notes,
                success=True,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we try alternatives
            last_error = exc
            continue

    return LoadResult(
        dataframe=pd.DataFrame(),
        encoding=encoding,
        delimiter=delimiter,
        header_row=header_arg,
        notes=notes,
        success=False,
        error=f"Failed to parse CSV after multiple attempts: {last_error}",
    )


def load_excel(raw_bytes: bytes, filename: str | None = None) -> LoadResult:
    """Loads an Excel workbook (.xlsx / .xls) into a DataFrame without crashing."""
    notes: list[str] = ["Loaded Excel workbook."]
    try:
        excel_file = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
        sheet_names = excel_file.sheet_names
        sheet_to_load = sheet_names[0] if sheet_names else 0
        if len(sheet_names) > 1:
            notes.append(
                f"Workbook has {len(sheet_names)} sheets ({', '.join(sheet_names[:3])}"
                + (f" and {len(sheet_names) - 3} more" if len(sheet_names) > 3 else "")
                + f"); loaded '{sheet_to_load}'."
            )

        df_raw = pd.read_excel(excel_file, sheet_name=sheet_to_load, header=None)
        if df_raw.empty:
            return LoadResult(
                dataframe=pd.DataFrame(),
                encoding="binary",
                delimiter="excel",
                header_row=None,
                notes=notes,
                success=True,
            )

        rows = [df_raw.iloc[i].tolist() for i in range(min(25, len(df_raw)))]
        best_header_row, has_header = _find_best_header_row(rows)

        if has_header:
            df = pd.read_excel(excel_file, sheet_name=sheet_to_load, header=best_header_row)
            header_arg = best_header_row
            if best_header_row > 0:
                notes.append(f"Auto-detected table header at row {best_header_row + 1} (skipped leading decorative/metadata rows).")
        else:
            df = df_raw.copy()
            df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
            header_arg = None
            notes.append("No explicit header row detected — generated column names instead.")

        df = _sanitize_dataframe_columns_and_cleanup(df)

        if len(df) > MAX_ROWS_FULL_PROCESSING:
            notes.append(
                f"Dataset exceeds {MAX_ROWS_FULL_PROCESSING:,} rows; sampled down for responsiveness."
            )
            df = df.sample(n=MAX_ROWS_FULL_PROCESSING, random_state=42).sort_index()

        return LoadResult(
            dataframe=df,
            encoding="binary",
            delimiter="excel",
            header_row=header_arg,
            notes=notes,
            success=True,
        )
    except Exception as exc:
        return LoadResult(
            dataframe=pd.DataFrame(),
            encoding="binary",
            delimiter="excel",
            header_row=None,
            notes=notes,
            success=False,
            error=f"Could not read Excel file: {exc}",
        )


def load_file(raw_bytes: bytes, filename: str | None = None) -> LoadResult:
    """Universal file loader: automatically routes between Excel (.xlsx, .xls)
    and CSV based on file extension and magic byte inspection."""
    safe_filename = sanitize_filename(filename)

    if not raw_bytes or len(raw_bytes) == 0:
        return LoadResult(
            dataframe=pd.DataFrame(),
            encoding="unknown",
            delimiter=",",
            header_row=None,
            notes=[],
            success=False,
            error="File is empty (0 bytes).",
        )

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return LoadResult(
            dataframe=pd.DataFrame(),
            encoding="unknown",
            delimiter=",",
            header_row=None,
            notes=[],
            success=False,
            error=f"File exceeds maximum allowed size ({len(raw_bytes) / (1024*1024):.1f}MB > {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB).",
        )

    is_excel = False
    if safe_filename.lower().endswith((".xlsx", ".xls")):
        is_excel = True
    elif raw_bytes.startswith(b"PK\x03\x04"):  # ZIP/XLSX magic signature
        is_excel = True

    if is_excel:
        return load_excel(raw_bytes, filename=safe_filename)
    return load_csv(raw_bytes, filename=safe_filename)

