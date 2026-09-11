"""
Mood Mentor - Integrated Report Generator
=========================================

Generates the final Mood Mentor report by combining:

Milestone 1:
    - Input text
    - Processed text
    - VADER sentiment
    - Sentiment scores

Milestone 2:
    - BERT emotion
    - BERT confidence
    - DistilBERT emotion
    - DistilBERT confidence
    - Multi-label emotions
    - Multi-label scores
    - Final emotion
    - Final confidence

The CSV keeps one row per analyzed input.
"""

import csv
import json
import os
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Milestone 1 report support
# ---------------------------------------------------------------------------

from text_ingestion import validate_input, IngestionError
from preprocessing import preprocess
from sentiment_analysis import analyze_sentiment


def build_summary(
    rows: List[Dict[str, Any]],
    num_submitted: int = None,
) -> Dict[str, Any]:
    """Build the Milestone 1 sentiment summary."""

    total = len(rows)

    if num_submitted is None:
        num_submitted = total

    skipped = max(0, num_submitted - total)

    positive_count = sum(
        1 for row in rows
        if row.get("sentiment_label") == "Positive"
    )

    negative_count = sum(
        1 for row in rows
        if row.get("sentiment_label") == "Negative"
    )

    neutral_count = sum(
        1 for row in rows
        if row.get("sentiment_label") == "Neutral"
    )

    return {
        "num_samples_submitted": num_submitted,
        "num_samples_analyzed": total,
        "num_samples_skipped_invalid": skipped,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_pct": round(
            100 * positive_count / total, 2
        ) if total else 0.0,
        "negative_pct": round(
            100 * negative_count / total, 2
        ) if total else 0.0,
        "neutral_pct": round(
            100 * neutral_count / total, 2
        ) if total else 0.0,
    }


def analyze_corpus(samples: List[str]) -> Dict[str, Any]:
    """
    Milestone 1 analysis.

    Runs:
        validation -> preprocessing -> VADER
    """

    rows = []

    for raw in samples:

        try:
            validated = validate_input(raw)

        except IngestionError:
            continue

        prep = preprocess(validated)
        sentiment = analyze_sentiment(prep.cleaned_text)

        rows.append({
            "input_text": prep.original_text,
            "processed_text": prep.processed_text,
            "sentiment_label": sentiment.label,
            "compound_score": round(sentiment.compound, 4),
            "positive_score": round(sentiment.pos, 4),
            "negative_score": round(sentiment.neg, 4),
            "neutral_score": round(sentiment.neu, 4),
        })

    return {
        "summary": build_summary(
            rows,
            num_submitted=len(samples),
        ),
        "rows": rows,
    }


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a report from already analyzed rows.

    Works for both:
        - Milestone 1 rows
        - Integrated Milestone 1 + Milestone 2 rows
    """

    return {
        "summary": build_summary(rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Milestone 2 integration
# ---------------------------------------------------------------------------

def add_milestone2_result(
    milestone1_row: Dict[str, Any],
    milestone2_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add Milestone 2 results to an existing Milestone 1 row.

    The original Milestone 1 fields are preserved.
    """

    emotion = milestone2_result.get("emotion", {})
    multi_label = milestone2_result.get("multi_label", {})
    final_emotion = milestone2_result.get("final_emotion", {})

    integrated_row = dict(milestone1_row)

    # -------------------------------------------------------
    # BERT
    # -------------------------------------------------------

    integrated_row["bert_emotion"] = (
        emotion.get("primary_emotion", "")
    )

    integrated_row["bert_confidence"] = (
        emotion.get("primary_confidence", "")
    )

    # -------------------------------------------------------
    # BERT all emotion scores
    # -------------------------------------------------------

    bert_scores = emotion.get("all_scores", {})

    for label in [
        "Joy",
        "Sadness",
        "Anger",
        "Fear",
        "Surprise",
        "Disgust",
    ]:
        integrated_row[f"bert_{label.lower()}_score"] = (
            bert_scores.get(label, "")
        )

    # -------------------------------------------------------
    # Multi-label
    # -------------------------------------------------------

    integrated_row["multilabel_emotions"] = "; ".join(
        multi_label.get("detected_emotions", [])
    )

    integrated_row["multilabel_primary_emotion"] = (
        multi_label.get("primary_emotion", "")
    )

    integrated_row["multilabel_threshold"] = (
        multi_label.get("threshold", "")
    )

    # -------------------------------------------------------
    # Multi-label emotion scores
    # -------------------------------------------------------

    multi_scores = multi_label.get("all_scores", {})

    for label in [
        "Joy",
        "Sadness",
        "Anger",
        "Fear",
        "Surprise",
        "Disgust",
    ]:
        integrated_row[
            f"multilabel_{label.lower()}_score"
        ] = multi_scores.get(label, "")

    # -------------------------------------------------------
    # Final emotion
    # -------------------------------------------------------

    integrated_row["final_emotion"] = (
        final_emotion.get("emotion", "")
    )

    integrated_row["final_confidence"] = (
        final_emotion.get("confidence", "")
    )

    integrated_row["final_status"] = (
        final_emotion.get("status", "")
    )

    return integrated_row


# ---------------------------------------------------------------------------
# CSV / JSON saving
# ---------------------------------------------------------------------------

def save_report_csv(
    report: Dict[str, Any],
    filepath: str,
) -> None:
    """Save report rows to CSV."""

    rows = report.get("rows", [])

    if not rows:
        fieldnames = [
            "input_text",
            "processed_text",
            "sentiment_label",
            "compound_score",
            "positive_score",
            "negative_score",
            "neutral_score",
        ]

    else:
        # Fixed ordering for the final integrated report.
        fieldnames = [
            # Milestone 1
            "input_text",
            "processed_text",
            "sentiment_label",
            "compound_score",
            "positive_score",
            "negative_score",
            "neutral_score",

            # BERT
            "bert_emotion",
            "bert_confidence",
            "bert_joy_score",
            "bert_sadness_score",
            "bert_anger_score",
            "bert_fear_score",
            "bert_surprise_score",
            "bert_disgust_score",

            # Multi-label
            "multilabel_emotions",
            "multilabel_primary_emotion",
            "multilabel_threshold",
            "multilabel_joy_score",
            "multilabel_sadness_score",
            "multilabel_anger_score",
            "multilabel_fear_score",
            "multilabel_surprise_score",
            "multilabel_disgust_score",

            # Final classification
            "final_emotion",
            "final_confidence",
            "final_status",
        ]

        # Add any unexpected fields at the end instead of losing data.
        existing_fields = set(fieldnames)

        for row in rows:
            for key in row.keys():
                if key not in existing_fields:
                    fieldnames.append(key)
                    existing_fields.add(key)

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_report_json(
    report: Dict[str, Any],
    filepath: str,
) -> None:
    """Save report as JSON."""

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_report(
    report: Dict[str, Any],
    out_name: str = "milestone1_report",
):
    """
    Save both CSV and JSON.

    Existing behavior is preserved:
        report.csv
        report.json
    """

    csv_path = f"{out_name}.csv"
    json_path = f"{out_name}.json"

    save_report_csv(
        report,
        csv_path,
    )

    save_report_json(
        report,
        json_path,
    )

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_report_summary(
    report: Dict[str, Any],
    csv_path: str = None,
    json_path: str = None,
    header: str = "MOOD MENTOR REPORT",
) -> None:
    """Print a compact report summary."""

    summary = report["summary"]

    print("\n" + "=" * 60)
    print(f" {header}")
    print("=" * 60)

    print(
        f" Samples analyzed : "
        f"{summary['num_samples_analyzed']}"
    )

    print(
        f" Positive         : "
        f"{summary['positive_count']} "
        f"({summary['positive_pct']}%)"
    )

    print(
        f" Negative         : "
        f"{summary['negative_count']} "
        f"({summary['negative_pct']}%)"
    )

    print(
        f" Neutral          : "
        f"{summary['neutral_count']} "
        f"({summary['neutral_pct']}%)"
    )

    if csv_path or json_path:

        print("-" * 60)

        if csv_path:
            print(
                f" Saved to: "
                f"{os.path.abspath(csv_path)}"
            )

        if json_path:
            print(
                f"           "
                f"{os.path.abspath(json_path)}"
            )

    print("=" * 60)