"""
Test a Saved Emotion Model with Your Own Text
=================================================
Task 1/2's "Test Model with Sample Text" step, but interactive: load
a saved BERT or DistilBERT model from disk and type in whatever
sentences you want, one at a time, and see the real prediction +
confidence scores for each - not just the 3 sentences hardcoded into
bert_model.py / distilbert_model.py.

Usage:
    python3 test_emotion_model.py                          # defaults to ./bert_emotion_model
    python3 test_emotion_model.py --model-dir ./bert_emotion_model
    python3 test_emotion_model.py --model-dir ./distilbert_emotion_model --model-type distilbert
"""

import argparse
import sys


def load_classifier(model_dir: str, model_type: str):
    if model_type == "bert":
        from bert_model import EmotionBERTClassifier
        return EmotionBERTClassifier.load(model_dir)
    else:
        from distilbert_model import EmotionDistilBERTClassifier
        return EmotionDistilBERTClassifier.load(model_dir)


def print_prediction(pred) -> None:
    print(f"\n  Primary emotion : {pred.primary_emotion}  (confidence {pred.primary_confidence})")
    print("  All scores:")
    # Sorted highest-to-lowest so the ranking is easy to read at a glance.
    for emotion, score in sorted(pred.all_scores.items(), key=lambda kv: kv[1], reverse=True):
        bar = "#" * int(score * 40)
        print(f"    {emotion:9s} {score:.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Interactively test a saved emotion model with your own text.")
    parser.add_argument("--model-dir", default="./bert_emotion_model", help="Path to the saved model folder")
    parser.add_argument("--model-type", choices=["bert", "distilbert"], default="bert")
    args = parser.parse_args()

    print(f"Loading {args.model_type} model from '{args.model_dir}' ...")
    try:
        classifier = load_classifier(args.model_dir, args.model_type)
    except Exception as e:
        print(f"\nCould not load the model: {e}")
        print("Check that --model-dir points at a folder created by classifier.save(), "
              "e.g. ./bert_emotion_model or ./distilbert_emotion_model.")
        sys.exit(1)

    print("Model loaded. Type a sentence to classify, or 'quit' to exit.\n")
    while True:
        text = input("Enter text: ").strip()
        if not text:
            print("(empty input - skipped)")
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        pred = classifier.predict(text)
        print_prediction(pred)
        print()


if __name__ == "__main__":
    main()