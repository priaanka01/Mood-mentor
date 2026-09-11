"""
Task 5 - Model Evaluation & Validation
===========================================
Evaluates a saved BERT/DistilBERT model against the held-out ISEAR
benchmark subset (isear_data.py's benchmark split - same seed/fraction
as training, so this is guaranteed to be rows the model never saw).
Computes accuracy, per-class precision/recall/F1, and macro F1 - every
number here comes from real prediction-vs-true-label counts, nothing
hardcoded (per the project's explicit "no hardcoded metrics" requirement).

Usage:
    # Evaluate one model:
    python3 evaluate_model.py --isear-csv ISEAR_spellchecked.csv --bert-dir ./bert_emotion_model

    # Evaluate and compare BOTH on the identical benchmark set:
    python3 evaluate_model.py --isear-csv ISEAR_spellchecked.csv --bert-dir ./bert_emotion_model --distilbert-dir ./distilbert_emotion_model

IMPORTANT: --benchmark-fraction and --seed must match whatever was
used during training's split (bert_model.py/distilbert_model.py both
use isear_data.py's defaults: fraction=0.15, seed=42) - otherwise this
would evaluate on a different split than the one actually held out,
silently invalidating the "never seen during training" guarantee.
"""

import argparse
from collections import defaultdict
from typing import Dict, List

from emotion_labels import EMOTION_LABELS
from isear_data import prepare_isear_dataset, EmotionRecord


def evaluate(classifier, benchmark_records: List[EmotionRecord]) -> Dict:
    """
    Runs the model's single-label prediction (softmax argmax, i.e.
    classifier.predict()) on every benchmark record and computes
    accuracy + per-class precision/recall/F1 + macro F1 from real
    true-positive/false-positive/false-negative counts.
    """
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)
    correct = 0
    total = len(benchmark_records)

    for record in benchmark_records:
        pred = classifier.predict(record.text)
        predicted_label = pred.primary_emotion
        true_label = record.label

        if predicted_label == true_label:
            correct += 1
            true_positive[true_label] += 1
        else:
            false_positive[predicted_label] += 1
            false_negative[true_label] += 1

    accuracy = correct / total if total else 0.0

    per_class = {}
    f1_scores = []
    for label in EMOTION_LABELS:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,  # how many real examples of this class existed in the benchmark
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "total_evaluated": total,
    }


def print_report(name: str, results: Dict) -> None:
    print(f"\n{'=' * 65}")
    print(f" {name} - EVALUATION RESULTS  (n={results['total_evaluated']} held-out ISEAR benchmark rows)")
    print(f"{'=' * 65}")
    print(f" Overall Accuracy : {results['accuracy']}")
    print(f" Macro F1-Score   : {results['macro_f1']}")
    print(f"{'-' * 65}")
    print(f" {'Emotion':10s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} {'Support':>8s}")
    for label, metrics in results["per_class"].items():
        print(f" {label:10s} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>8.4f} {metrics['support']:>8d}")


def main():
    parser = argparse.ArgumentParser(description="Task 5 - Evaluate BERT/DistilBERT against held-out ISEAR benchmark")
    parser.add_argument("--isear-csv", required=True, help="Path to the ISEAR CSV (same file used for training)")
    parser.add_argument("--bert-dir", default=None, help="Path to a saved BERT model to evaluate")
    parser.add_argument("--distilbert-dir", default=None, help="Path to a saved DistilBERT model to evaluate")
    parser.add_argument(
        "--benchmark-fraction", type=float, default=0.15,
        help="Must match the fraction used during training's split (default 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Must match training's split seed (default 42)")
    args = parser.parse_args()

    if not args.bert_dir and not args.distilbert_dir:
        parser.error("Provide at least one of --bert-dir or --distilbert-dir")

    _train_records, benchmark_records = prepare_isear_dataset(
        args.isear_csv, benchmark_fraction=args.benchmark_fraction, seed=args.seed
    )
    print(f"\nEvaluating against {len(benchmark_records)} held-out benchmark rows (never used in training).")

    results_summary = {}

    if args.bert_dir:
        print(f"\nLoading BERT from '{args.bert_dir}' ...")
        from bert_model import EmotionBERTClassifier
        bert_classifier = EmotionBERTClassifier.load(args.bert_dir)
        bert_results = evaluate(bert_classifier, benchmark_records)
        print_report("BERT", bert_results)
        results_summary["BERT"] = bert_results

    if args.distilbert_dir:
        print(f"\nLoading DistilBERT from '{args.distilbert_dir}' ...")
        from distilbert_model import EmotionDistilBERTClassifier
        distil_classifier = EmotionDistilBERTClassifier.load(args.distilbert_dir)
        distil_results = evaluate(distil_classifier, benchmark_records)
        print_report("DistilBERT", distil_results)
        results_summary["DistilBERT"] = distil_results

    if len(results_summary) == 2:
        print(f"\n{'=' * 65}")
        print(" HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 65}")
        for name, res in results_summary.items():
            print(f" {name:12s} Accuracy={res['accuracy']:.4f}  Macro F1={res['macro_f1']:.4f}")
        better = max(results_summary.items(), key=lambda kv: kv[1]["macro_f1"])
        print(f"\n Better performer (by macro F1): {better[0]} ({better[1]['macro_f1']})")


if __name__ == "__main__":
    main()