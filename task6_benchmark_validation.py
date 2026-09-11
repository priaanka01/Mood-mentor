"""
Task 6 - ISEAR Benchmark Validation
========================================
Builds on evaluate_model.py's held-out ISEAR benchmark (same split,
same seed - never seen during training) to specifically satisfy
Task 6's checklist:
  - Emotion prediction accuracy       (reused from evaluate_model.py's logic)
  - Incorrect predictions             (full list + confidence scores)
  - Emotion-wise performance          (confusion matrix: what gets
                                        confused with what, not just
                                        per-class P/R/F1)
  - Confidence scores                 (shown per incorrect prediction,
                                        including the most confidently
                                        WRONG cases - the concerning ones)
  - Model consistency                 (predicting the same text twice
                                        should give an identical result
                                        - the model has no dropout/
                                        randomness active in eval mode,
                                        so this should always pass; a
                                        failure here would mean
                                        something is wrong with how
                                        the model is being loaded/run)
  - Compare predicted vs expected     (confusion matrix does this directly)
  - Identify major classification
    issues                            (printed as a written summary of
                                        the biggest confusion pairs)

Usage:
    python3 task6_benchmark_validation.py --isear-csv ISEAR_spellchecked.csv --model-dir ./bert_emotion_model
    python3 task6_benchmark_validation.py --isear-csv ISEAR_spellchecked.csv --model-dir ./distilbert_emotion_model --model-type distilbert
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

from emotion_labels import EMOTION_LABELS
from isear_data import prepare_isear_dataset, EmotionRecord


def load_classifier(model_dir: str, model_type: str):
    if model_type == "bert":
        from bert_model import EmotionBERTClassifier
        return EmotionBERTClassifier.load(model_dir)
    else:
        from distilbert_model import EmotionDistilBERTClassifier
        return EmotionDistilBERTClassifier.load(model_dir)


def run_validation(classifier, benchmark_records: List[EmotionRecord]) -> Dict:
    """
    Single pass over the benchmark set (avoids re-running the model
    twice for metrics + diagnostics): builds the confusion matrix,
    collects every incorrect prediction with its confidence, and
    computes accuracy - all from real predictions on real held-out data.
    """
    # confusion[true_label][predicted_label] = count
    confusion = {t: {p: 0 for p in EMOTION_LABELS} for t in EMOTION_LABELS}
    misclassified: List[Tuple[str, str, str, float]] = []  # (text, true, predicted, confidence)
    correct = 0

    for record in benchmark_records:
        pred = classifier.predict(record.text)
        predicted_label = pred.primary_emotion
        true_label = record.label

        confusion[true_label][predicted_label] += 1

        if predicted_label == true_label:
            correct += 1
        else:
            misclassified.append((record.text, true_label, predicted_label, pred.primary_confidence))

    accuracy = correct / len(benchmark_records) if benchmark_records else 0.0
    return {"confusion": confusion, "misclassified": misclassified, "accuracy": round(accuracy, 4)}


def check_consistency(classifier, sample_texts: List[str]) -> bool:
    """
    Runs predict() twice on each of a small sample of texts and
    confirms identical output both times. The model is in eval mode
    (no dropout), so this should always pass - a failure would signal
    something wrong in how the model is loaded/run, not normal noise.
    """
    for text in sample_texts:
        first = classifier.predict(text)
        second = classifier.predict(text)
        if first.primary_emotion != second.primary_emotion or first.all_scores != second.all_scores:
            print(f"  INCONSISTENT on: '{text[:60]}...'")
            print(f"    Run 1: {first.primary_emotion} {first.all_scores}")
            print(f"    Run 2: {second.primary_emotion} {second.all_scores}")
            return False
    return True


def print_confusion_matrix(confusion: Dict) -> None:
    header_label = "True \\ Pred"
    print(f"\n{header_label:12s}", end="")
    for label in EMOTION_LABELS:
        print(f"{label[:8]:>9s}", end="")
    print()
    for true_label in EMOTION_LABELS:
        print(f"{true_label:12s}", end="")
        for pred_label in EMOTION_LABELS:
            count = confusion[true_label][pred_label]
            print(f"{count:>9d}", end="")
        print()


def summarize_top_confusions(confusion: Dict, top_n: int = 5) -> List[Tuple[str, str, int]]:
    """Biggest off-diagonal (true != predicted) confusion pairs, most common first."""
    pairs = []
    for true_label in EMOTION_LABELS:
        for pred_label in EMOTION_LABELS:
            if true_label != pred_label and confusion[true_label][pred_label] > 0:
                pairs.append((true_label, pred_label, confusion[true_label][pred_label]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Task 6 - ISEAR benchmark validation")
    parser.add_argument("--isear-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-type", choices=["bert", "distilbert"], default="bert")
    parser.add_argument("--benchmark-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-misclassified", type=int, default=15, help="How many incorrect examples to print (default 15)")
    args = parser.parse_args()

    _train_records, benchmark_records = prepare_isear_dataset(
        args.isear_csv, benchmark_fraction=args.benchmark_fraction, seed=args.seed
    )
    print(f"\nValidating against {len(benchmark_records)} held-out ISEAR benchmark rows.")

    print(f"\nLoading {args.model_type} from '{args.model_dir}' ...")
    classifier = load_classifier(args.model_dir, args.model_type)

    print("\n" + "=" * 65)
    print(" MODEL CONSISTENCY CHECK")
    print("=" * 65)
    sample_for_consistency = [r.text for r in benchmark_records[:10]]
    consistent = check_consistency(classifier, sample_for_consistency)
    print(f"Consistency check (10 samples, predicted twice each): {'PASS - identical every time' if consistent else 'FAIL - see above'}")

    print("\n" + "=" * 65)
    print(" RUNNING FULL BENCHMARK VALIDATION")
    print("=" * 65)
    results = run_validation(classifier, benchmark_records)
    print(f"\nOverall accuracy: {results['accuracy']}  ({len(results['misclassified'])} incorrect out of {len(benchmark_records)})")

    print("\n" + "=" * 65)
    print(" CONFUSION MATRIX (rows = true label, columns = predicted)")
    print("=" * 65)
    print_confusion_matrix(results["confusion"])

    print("\n" + "=" * 65)
    print(" TOP CONFUSION PAIRS (major classification issues)")
    print("=" * 65)
    top_confusions = summarize_top_confusions(results["confusion"])
    for true_label, pred_label, count in top_confusions:
        print(f"  True '{true_label}' predicted as '{pred_label}': {count} times")

    print("\n" + "=" * 65)
    print(f" SAMPLE INCORRECT PREDICTIONS (showing up to {args.show_misclassified}, highest-confidence wrong ones first)")
    print("=" * 65)
    # Sort by confidence descending: a WRONG prediction the model was
    # very confident about is more concerning than a low-confidence miss.
    sorted_wrong = sorted(results["misclassified"], key=lambda x: x[3], reverse=True)
    for text, true_label, predicted_label, confidence in sorted_wrong[: args.show_misclassified]:
        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"  [{confidence:.4f}] True={true_label:9s} Predicted={predicted_label:9s} | {preview}")


if __name__ == "__main__":
    main()