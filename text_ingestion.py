"""
Task 1 - Text Ingestion Module
================================
Handles all supported input methods for the Mood Mentor pipeline:
  - Direct text input (string)
  - .txt file upload
  - .csv file upload (expects a text-bearing column)

Every entry point funnels into `validate_input()` before being handed
to preprocessing, and invalid/empty input is rejected with a clear
IngestionError rather than silently passed downstream.
"""

import os
import csv
from dataclasses import dataclass
from typing import List


class IngestionError(Exception):
    """Raised when input text/file fails validation."""
    pass


@dataclass
class IngestedRecord:
    source: str          # 'text_input' | 'txt_file' | 'csv_file'
    origin: str           # filename or 'direct_input'
    raw_text: str


def validate_input(text: str) -> str:
    """
    Validates a single piece of raw text before it can proceed to
    preprocessing.

    Rules:
      - Must not be None
      - Must not be empty / whitespace-only after stripping
      - Must be a string type
      - Must contain at least one alphanumeric character
        (rejects pure punctuation/noise-only input)

    Returns the stripped text on success.
    Raises IngestionError on failure.
    """
    if text is None:
        raise IngestionError("Input is None.")
    if not isinstance(text, str):
        raise IngestionError(f"Input must be a string, got {type(text).__name__}.")

    stripped = text.strip()
    if stripped == "":
        raise IngestionError("Input text is empty or whitespace-only.")

    if not any(ch.isalnum() for ch in stripped):
        raise IngestionError("Input contains no usable (alphanumeric) content.")

    return stripped


def ingest_text_input(text: str) -> IngestedRecord:
    """Task 1 flow step: Create/enter text input."""
    validated = validate_input(text)
    return IngestedRecord(source="text_input", origin="direct_input", raw_text=validated)


def ingest_txt_file(filepath: str) -> IngestedRecord:
    """Task 1 flow step: Upload .txt file."""
    if not os.path.exists(filepath):
        raise IngestionError(f".txt file not found: {filepath}")
    if not filepath.lower().endswith(".txt"):
        raise IngestionError(f"Expected a .txt file, got: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    validated = validate_input(content)
    return IngestedRecord(source="txt_file", origin=os.path.basename(filepath), raw_text=validated)


def ingest_csv_file(filepath: str, text_column: str = None) -> List[IngestedRecord]:
    """
    Task 1 flow step: Upload .csv file.

    If `text_column` is not given, the module tries common column
    names ('text', 'review', 'comment', 'message', 'content') and
    falls back to the first column in the file.

    Returns a list of IngestedRecord, one per valid row. Rows that
    fail validation are skipped (not silently merged into good data),
    and a summary is available via the returned list length vs. row
    count if the caller wants to log discrepancies.
    """
    if not os.path.exists(filepath):
        raise IngestionError(f".csv file not found: {filepath}")
    if not filepath.lower().endswith(".csv"):
        raise IngestionError(f"Expected a .csv file, got: {filepath}")

    records = []
    candidate_cols = ["text", "review", "comment", "message", "content"]

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise IngestionError("CSV file has no header row / is empty.")

        col = text_column
        if col is None:
            for c in candidate_cols:
                if c in reader.fieldnames:
                    col = c
                    break
            if col is None:
                col = reader.fieldnames[0]

        if col not in reader.fieldnames:
            raise IngestionError(f"Column '{col}' not found in CSV. Available: {reader.fieldnames}")

        for i, row in enumerate(reader):
            raw = row.get(col, "")
            try:
                validated = validate_input(raw)
                records.append(
                    IngestedRecord(
                        source="csv_file",
                        origin=f"{os.path.basename(filepath)}:row{i+2}",  # +2 = header + 1-index
                        raw_text=validated,
                    )
                )
            except IngestionError:
                # skip invalid row, keep ingesting the rest
                continue

    if not records:
        raise IngestionError("No valid text rows found in CSV after validation.")

    return records
