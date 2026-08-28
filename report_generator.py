"""
Task 4 - Initial Emotion/Sentiment Report Generator
======================================================
Runs the full pipeline (ingestion -> preprocessing -> VADER) across a
sample text corpus and produces a structured report:

  - input_text
  - processed_text
  - sentiment classification
  - sentiment scores (compound/pos/neg/neu)
  - aggregate positive/negative/neutral counts
  - number of analyzed samples

The report is returned as a list of dict rows (JSON-friendly) plus a
summary dict, and can be written to CSV/JSON for milestone sign-off.
"""

import csv
import json
from typing import List, Dict, Any

from text_ingestion import validate_input, IngestionError
from preprocessing import preprocess
from sentiment_analysis import analyze_sentiment


def build_summary(rows: List[Dict[str, Any]], num_submitted: int = None) -> Dict[str, Any]:
    """
    Single source of truth for turning a list of already-analyzed row
    dicts (each must have a 'sentiment_label' key) into the aggregate
    summary block: counts, percentages, and sample totals.

    Used by BOTH `analyze_corpus` (Task 4, raw-string input) and the
    CLI's csv/txt/text pipeline runs (Task 5, PipelineResult input),
    so the counting logic exists in exactly one place.

    `num_submitted` lets a caller report how many samples came in
    before any were skipped for being invalid; if omitted, it's
    assumed to equal len(rows) (nothing was skipped).
    """
    total = len(rows)
    if num_submitted is None:
        num_submitted = total
    skipped = num_submitted - total

    pos_count = sum(1 for r in rows if r["sentiment_label"] == "Positive")
    neg_count = sum(1 for r in rows if r["sentiment_label"] == "Negative")
    neu_count = sum(1 for r in rows if r["sentiment_label"] == "Neutral")

    return {
        "num_samples_submitted": num_submitted,
        "num_samples_analyzed": total,
        "num_samples_skipped_invalid": skipped,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_pct": round(100 * pos_count / total, 2) if total else 0.0,
        "negative_pct": round(100 * neg_count / total, 2) if total else 0.0,
        "neutral_pct": round(100 * neu_count / total, 2) if total else 0.0,
    }


def analyze_corpus(samples: List[str]) -> Dict[str, Any]:
    """
    Task 4 core routine. Takes a list of raw text samples, runs each
    through ingestion validation -> preprocessing -> VADER, and builds
    a full report.
    """
    rows = []
    skipped = 0

    for raw in samples:
        try:
            validated = validate_input(raw)
        except IngestionError:
            skipped += 1
            continue

        prep = preprocess(validated)
        sent = analyze_sentiment(prep.cleaned_text)

        rows.append({
            "input_text": prep.original_text,
            "processed_text": prep.processed_text,
            "sentiment_label": sent.label,
            "compound_score": sent.compound,
            "positive_score": sent.pos,
            "negative_score": sent.neg,
            "neutral_score": sent.neu,
        })

    summary = build_summary(rows, num_submitted=len(samples))
    return {"summary": summary, "rows": rows}


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task 5 / CLI helper: wrap already-computed row dicts (e.g. from
    PipelineResult.as_dict() in pipeline.py) into the same
    {"summary": ..., "rows": ...} report shape as `analyze_corpus`,
    using the same `build_summary` counting logic. No invalid rows
    to skip here since ingestion already filtered them upstream.
    """
    return {"summary": build_summary(rows), "rows": rows}


def save_report_csv(report: Dict[str, Any], filepath: str) -> None:
    rows = report["rows"]
    if not rows:
        # still write an empty file with headers for traceability
        fieldnames = ["input_text", "processed_text", "sentiment_label",
                      "compound_score", "positive_score", "negative_score", "neutral_score"]
    else:
        fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_report_json(report: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)