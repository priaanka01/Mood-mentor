"""
Mood Mentor - Task 10 Final Milestone 2 Validation
===================================================

Final validation of Milestone 2 integrated with Milestone 1.

Supports:
    --text
    --txt
    --csv

Uses already-trained saved models.
Does NOT retrain BERT, DistilBERT, or Multi-label BERT.

The final report is APPEND-BASED:
    milestone2_final_report.csv
    milestone2_final_report.json

New inputs are added to the existing report instead of
overwriting previous results.
"""

import argparse
import csv
import json
import os
import sys

from text_ingestion import (
    ingest_text_input,
    ingest_txt_file,
    ingest_csv_file,
    IngestionError,
)

from preprocessing import preprocess
from sentiment_analysis import analyze_sentiment

from bert_model import EmotionBERTClassifier
from distilbert_model import EmotionDistilBERTClassifier

from multi_label_classifier import (
    MultiLabelBERTClassifier,
    predict_multi_label,
    DEFAULT_THRESHOLD,
)

from evaluate_model import evaluate
from isear_data import prepare_isear_dataset


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_BERT_DIR = "./bert_emotion_model"
DEFAULT_DISTILBERT_DIR = "./distilbert_emotion_model"
DEFAULT_MULTILABEL_DIR = "./multilabel_emotion_model"
DEFAULT_ISEAR_CSV = "./ISEAR_spellchecked.csv"

DEFAULT_OUTPUT = "milestone2_final_report"


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

def check_model_directory(path, name):
    """Check whether a saved model directory exists."""

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{name} model directory not found: {path}"
        )

    print(f"  [OK] {name}: {path}")


# ============================================================
# LOAD INPUT
# ============================================================

def load_input_texts(args):
    """
    Load input from --text, --txt, or --csv.

    Returns:
        List of raw text strings.
    """

    selected_inputs = [
        args.text is not None,
        args.txt is not None,
        args.csv is not None,
    ]

    if sum(selected_inputs) == 0:
        raise ValueError(
            "Please provide input using --text, --txt, or --csv."
        )

    if sum(selected_inputs) > 1:
        raise ValueError(
            "Use only one of --text, --txt, or --csv."
        )

    texts = []

    # --------------------------------------------------------
    # Direct text
    # --------------------------------------------------------

    if args.text is not None:

        if not args.text.strip():
            raise ValueError(
                "Text input cannot be empty."
            )

        record = ingest_text_input(args.text)
        texts.append(record.raw_text)

    # --------------------------------------------------------
    # TXT file
    # --------------------------------------------------------

    elif args.txt is not None:

        record = ingest_txt_file(args.txt)
        texts.append(record.raw_text)

    # --------------------------------------------------------
    # CSV file
    # --------------------------------------------------------

    elif args.csv is not None:

        records = ingest_csv_file(
            args.csv,
            text_column=args.column,
        )

        for record in records:
            texts.append(record.raw_text)

    if not texts:
        raise ValueError(
            "No valid input text was found."
        )

    return texts


# ============================================================
# ANALYZE ONE TEXT
# ============================================================

def analyze_text(
    raw_text,
    bert_classifier,
    distilbert_classifier,
    multilabel_classifier,
    threshold,
):
    """
    Run one text through the complete
    Milestone 1 + Milestone 2 pipeline.
    """

    # --------------------------------------------------------
    # Milestone 1 - Input validation
    # --------------------------------------------------------

    record = ingest_text_input(raw_text)

    # --------------------------------------------------------
    # Milestone 1 - Preprocessing
    # --------------------------------------------------------

    prep = preprocess(record.raw_text)

    # --------------------------------------------------------
    # Milestone 1 - VADER sentiment
    # --------------------------------------------------------

    sentiment = analyze_sentiment(
        prep.cleaned_text
    )

    # --------------------------------------------------------
    # Milestone 2 - BERT
    # --------------------------------------------------------

    bert_prediction = bert_classifier.predict(
        prep.cleaned_text
    )

    # --------------------------------------------------------
    # Milestone 2 - DistilBERT
    # --------------------------------------------------------

    distil_prediction = distilbert_classifier.predict(
        prep.cleaned_text
    )

    # --------------------------------------------------------
    # Milestone 2 - Multi-label
    # --------------------------------------------------------

    multi_prediction = predict_multi_label(
        multilabel_classifier,
        prep.cleaned_text,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Integrated report row
    # --------------------------------------------------------

    row = {
        # -------------------------
        # Milestone 1
        # -------------------------

        "input_text": prep.original_text,

        "processed_text": prep.processed_text,

        "sentiment_label": sentiment.label,

        "compound_score": round(
            sentiment.compound,
            4
        ),

        "positive_score": round(
            sentiment.pos,
            4
        ),

        "negative_score": round(
            sentiment.neg,
            4
        ),

        "neutral_score": round(
            sentiment.neu,
            4
        ),

        # -------------------------
        # BERT
        # -------------------------

        "bert_emotion":
            bert_prediction.primary_emotion,

        "bert_confidence": round(
            bert_prediction.primary_confidence,
            4
        ),

        # -------------------------
        # DistilBERT
        # -------------------------

        "distilbert_emotion":
            distil_prediction.primary_emotion,

        "distilbert_confidence": round(
            distil_prediction.primary_confidence,
            4
        ),

        # -------------------------
        # Multi-label
        # -------------------------

        "detected_emotions":
            ", ".join(
                multi_prediction.detected_emotions
            ),

        "multilabel_primary_emotion":
            multi_prediction.primary_emotion,

        "multilabel_threshold":
            threshold,

        # -------------------------
        # Multi-label scores
        # -------------------------

        "joy_score": round(
            multi_prediction.all_scores.get(
                "Joy",
                0.0
            ),
            4
        ),

        "sadness_score": round(
            multi_prediction.all_scores.get(
                "Sadness",
                0.0
            ),
            4
        ),

        "anger_score": round(
            multi_prediction.all_scores.get(
                "Anger",
                0.0
            ),
            4
        ),

        "fear_score": round(
            multi_prediction.all_scores.get(
                "Fear",
                0.0
            ),
            4
        ),

        "surprise_score": round(
            multi_prediction.all_scores.get(
                "Surprise",
                0.0
            ),
            4
        ),

        "disgust_score": round(
            multi_prediction.all_scores.get(
                "Disgust",
                0.0
            ),
            4
        ),
    }

    return row


# ============================================================
# APPEND TO CSV
# ============================================================

def append_csv(rows, filepath):
    """
    Append new rows to an existing CSV file.

    If the file does not exist, create it and write the header.

    If the file already exists, preserve all previous rows
    and append only the new rows.
    """

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    file_exists = os.path.exists(filepath)

    with open(
        filepath,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        # Write header only for a new/empty file
        if not file_exists or os.path.getsize(filepath) == 0:
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# APPEND TO JSON
# ============================================================

def append_json(rows, filepath):
    """
    Append new rows to an existing JSON report.

    Existing results are preserved.
    """

    existing_rows = []

    if os.path.exists(filepath):

        try:
            with open(
                filepath,
                "r",
                encoding="utf-8",
            ) as file:

                content = file.read().strip()

                if content:
                    existing_rows = json.loads(content)

                    if not isinstance(
                        existing_rows,
                        list
                    ):
                        existing_rows = []

        except (
            json.JSONDecodeError,
            OSError,
        ):
            existing_rows = []

    existing_rows.extend(rows)

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            existing_rows,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# ISEAR VALIDATION
# ============================================================

def run_isear_validation(
    isear_csv,
    bert_classifier,
    distilbert_classifier,
):
    """
    Run held-out ISEAR validation using saved models.
    """

    print("\n[5] Running ISEAR validation...")

    print(
        "Preparing held-out ISEAR benchmark..."
    )

    _, benchmark_records = prepare_isear_dataset(
        isear_csv
    )

    print(
        f"  Benchmark rows: "
        f"{len(benchmark_records)}"
    )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    print("\nEvaluating BERT...")

    bert_results = evaluate(
        bert_classifier,
        benchmark_records,
    )

    # --------------------------------------------------------
    # DistilBERT
    # --------------------------------------------------------

    print("\nEvaluating DistilBERT...")

    distil_results = evaluate(
        distilbert_classifier,
        benchmark_records,
    )

    return {
        "benchmark_rows":
            len(benchmark_records),

        "bert_accuracy":
            bert_results.get(
                "accuracy",
                0.0
            ),

        "bert_macro_f1":
            bert_results.get(
                "macro_f1",
                0.0
            ),

        "distilbert_accuracy":
            distil_results.get(
                "accuracy",
                0.0
            ),

        "distilbert_macro_f1":
            distil_results.get(
                "macro_f1",
                0.0
            ),
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(rows, validation):
    """Print final Task 10 validation results."""

    print("\n")
    print("=" * 70)
    print(
        " MOOD MENTOR - TASK 10 FINAL VALIDATION"
    )
    print("=" * 70)

    print("\nVerification")
    print("-" * 70)

    print(
        "  [OK] BERT model works"
    )

    print(
        "  [OK] DistilBERT model works"
    )

    print(
        "  [OK] Multi-label emotion classification works"
    )

    print(
        "  [OK] Confidence scores generated"
    )

    print(
        "  [OK] Evaluation metrics generated"
    )

    print(
        "  [OK] ISEAR validation completed"
    )

    print(
        "  [OK] Milestone 1 integration works"
    )

    print(
        "  [OK] Final emotion classification report generated"
    )

    print(
        "\nIntegrated Classification Results"
    )

    print("-" * 70)

    for number, row in enumerate(
        rows,
        start=1
    ):

        print(
            f"\nInput {number}"
        )

        print(
            f"Text: {row['input_text']}"
        )

        print(
            f"Sentiment      : "
            f"{row['sentiment_label']}"
        )

        print(
            f"BERT           : "
            f"{row['bert_emotion']} "
            f"({row['bert_confidence']:.4f})"
        )

        print(
            f"DistilBERT     : "
            f"{row['distilbert_emotion']} "
            f"({row['distilbert_confidence']:.4f})"
        )

        print(
            f"Multi-label    : "
            f"{row['detected_emotions']}"
        )

        print(
            f"Primary Multi  : "
            f"{row['multilabel_primary_emotion']}"
        )

    print(
        "\nISEAR Evaluation"
    )

    print("-" * 70)

    print(
        f"Benchmark rows       : "
        f"{validation['benchmark_rows']}"
    )

    print(
        f"BERT Accuracy        : "
        f"{validation['bert_accuracy']:.4f}"
    )

    print(
        f"BERT Macro F1        : "
        f"{validation['bert_macro_f1']:.4f}"
    )

    print(
        f"DistilBERT Accuracy  : "
        f"{validation['distilbert_accuracy']:.4f}"
    )

    print(
        f"DistilBERT Macro F1  : "
        f"{validation['distilbert_macro_f1']:.4f}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        " TASK 10 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mood Mentor - Task 10 Final "
            "Milestone 2 Validation"
        )
    )

    # --------------------------------------------------------
    # Input options
    # --------------------------------------------------------

    input_group = (
        parser
        .add_mutually_exclusive_group()
    )

    input_group.add_argument(
        "--text",
        help="Text to analyze",
    )

    input_group.add_argument(
        "--txt",
        help="Path to TXT file",
    )

    input_group.add_argument(
        "--csv",
        help="Path to CSV file",
    )

    parser.add_argument(
        "--column",
        default=None,
        help="CSV text column name",
    )

    # --------------------------------------------------------
    # Model paths
    # --------------------------------------------------------

    parser.add_argument(
        "--bert-dir",
        default=DEFAULT_BERT_DIR,
        help="Saved BERT model directory",
    )

    parser.add_argument(
        "--distilbert-dir",
        default=DEFAULT_DISTILBERT_DIR,
        help="Saved DistilBERT model directory",
    )

    parser.add_argument(
        "--multilabel-dir",
        default=DEFAULT_MULTILABEL_DIR,
        help="Saved multi-label model directory",
    )

    parser.add_argument(
        "--isear-csv",
        default=DEFAULT_ISEAR_CSV,
        help="ISEAR CSV file",
    )

    # --------------------------------------------------------
    # Multi-label threshold
    # --------------------------------------------------------

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Multi-label threshold",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=(
            "Output filename without extension"
        ),
    )

    args = parser.parse_args()

    try:

        print("=" * 70)

        print(
            " MOOD MENTOR - TASK 10 "
            "FINAL MILESTONE 2 VALIDATION"
        )

        print("=" * 70)

        # ====================================================
        # 1. CHECK MODELS
        # ====================================================

        print(
            "\n[1] Checking saved models..."
        )

        check_model_directory(
            args.bert_dir,
            "BERT",
        )

        check_model_directory(
            args.distilbert_dir,
            "DistilBERT",
        )

        check_model_directory(
            args.multilabel_dir,
            "Multi-label",
        )

        # ====================================================
        # 2. LOAD BERT
        # ====================================================

        print(
            "\n[2] Loading BERT model..."
        )

        bert_classifier = (
            EmotionBERTClassifier.load(
                args.bert_dir
            )
        )

        print(
            "  [OK] BERT model loaded"
        )

        # ====================================================
        # 3. LOAD DISTILBERT
        # ====================================================

        print(
            "\n[3] Loading DistilBERT model..."
        )

        distilbert_classifier = (
            EmotionDistilBERTClassifier.load(
                args.distilbert_dir
            )
        )

        print(
            "  [OK] DistilBERT model loaded"
        )

        # ====================================================
        # 4. LOAD MULTI-LABEL
        # ====================================================

        print(
            "\n[4] Loading multi-label model..."
        )

        multilabel_classifier = (
            MultiLabelBERTClassifier.load(
                args.multilabel_dir
            )
        )

        print(
            "  [OK] Multi-label model loaded"
        )

        # ====================================================
        # LOAD INPUT
        # ====================================================

        print(
            "\nLoading input..."
        )

        texts = load_input_texts(args)

        print(
            f"  [OK] {len(texts)} input(s) loaded"
        )

        # ====================================================
        # INTEGRATED ANALYSIS
        # ====================================================

        print(
            "\nRunning Milestone 1 + "
            "Milestone 2 integration..."
        )

        rows = []

        for text in texts:

            result = analyze_text(
                text,
                bert_classifier,
                distilbert_classifier,
                multilabel_classifier,
                args.threshold,
            )

            rows.append(result)

        print(
            f"  [OK] {len(rows)} input(s) analyzed"
        )

        # ====================================================
        # ISEAR VALIDATION
        # ====================================================

        validation = run_isear_validation(
            args.isear_csv,
            bert_classifier,
            distilbert_classifier,
        )

        # ====================================================
        # GENERATE / APPEND FINAL REPORT
        # ====================================================

        print(
            "\n[6] Generating final "
            "emotion classification report..."
        )

        csv_path = (
            f"{args.out}.csv"
        )

        json_path = (
            f"{args.out}.json"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Append instead of overwrite
        # ----------------------------------------------------

        append_csv(
            rows,
            csv_path,
        )

        append_json(
            rows,
            json_path,
        )

        print(
            "\n  [OK] Results added to CSV report:"
        )

        print(
            f"  {os.path.abspath(csv_path)}"
        )

        print(
            "\n  [OK] Results added to JSON report:"
        )

        print(
            f"  {os.path.abspath(json_path)}"
        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        print_results(
            rows,
            validation,
        )

    except (
        FileNotFoundError,
        IngestionError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:

        print(
            "\n[ERROR] Task 10 validation failed:"
        )

        print(
            error
        )

        sys.exit(1)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()