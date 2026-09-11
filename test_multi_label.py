"""
Task 3 Verification - Demo + Interactive Multi-Label Tester
=================================================================
Task 3 explicitly asks to test with: single emotion, multiple
emotions, strong emotional expressions, and mixed emotions. This
script runs one built-in example of each automatically, then drops
into an interactive loop (same pattern as test_emotion_model.py) so
you can try your own text too.

Usage:
    python3 test_multi_label.py --model-dir ./bert_emotion_model
    python3 test_multi_label.py --model-dir ./distilbert_emotion_model --model-type distilbert
    python3 test_multi_label.py --model-dir ./bert_emotion_model --threshold 0.3
"""

import argparse

from multi_label_classifier import predict_multi_label, DEFAULT_THRESHOLD

# One built-in example per category Task 3 explicitly asks to test.
DEMO_TEXTS = {
    "Single emotion": "I am so happy about my promotion today.",
    "Multiple emotions": "I am excited about the new opportunity but nervous about the outcome.",
    "Strong emotional expression": "I am absolutely furious - this is the worst betrayal I have ever experienced.",
    "Mixed emotions": (
        "I was relieved the project was finally over, but sad to say goodbye to the "
        "team, and honestly a little scared about what comes next."
    ),
}


def load_classifier(model_dir: str, model_type: str):
    if model_type == "bert":
        from bert_model import EmotionBERTClassifier
        return EmotionBERTClassifier.load(model_dir)
    else:
        from distilbert_model import EmotionDistilBERTClassifier
        return EmotionDistilBERTClassifier.load(model_dir)


def print_result(pred) -> None:
    print(f"  Detected emotion(s): {', '.join(pred.detected_emotions)}")
    print(f"  Primary emotion    : {pred.primary_emotion}")
    print("  All scores (independent per-class, not softmax - won't sum to 1):")
    for emotion, score in sorted(pred.all_scores.items(), key=lambda kv: kv[1], reverse=True):
        bar = "#" * int(score * 40)
        print(f"    {emotion:9s} {score:.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Task 3 - multi-label emotion classification demo/tester")
    parser.add_argument("--model-dir", default="./bert_emotion_model")
    parser.add_argument("--model-type", choices=["bert", "distilbert"], default="bert")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Score cutoff for an emotion to count as 'detected' (default {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    print(f"Loading {args.model_type} model from '{args.model_dir}' ...")
    classifier = load_classifier(args.model_dir, args.model_type)
    print("Model loaded.\n")

    print("=" * 65)
    print(" TASK 3 DEMO - one example per required test category")
    print("=" * 65)
    for category, text in DEMO_TEXTS.items():
        print(f"\n[{category}]")
        print(f"Text: {text}")
        pred = predict_multi_label(classifier, text, threshold=args.threshold)
        print_result(pred)

    print("\n" + "=" * 65)
    print(" INTERACTIVE MODE - type your own text, 'quit' to exit")
    print("=" * 65)
    while True:
        text = input("\nEnter text: ").strip()
        if not text:
            print("(empty input - skipped)")
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        pred = predict_multi_label(classifier, text, threshold=args.threshold)
        print_result(pred)


if __name__ == "__main__":
    main()