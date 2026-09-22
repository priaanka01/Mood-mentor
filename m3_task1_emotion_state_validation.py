"""
Milestone 3 - Task 1 validation.

Checks that intensity, polarity, mixed-state detection and severity
are calculated dynamically for different inputs.
"""

import argparse
from m3_emotion_state import analyze_emotional_state
from multi_label_classifier import MultiLabelBERTClassifier


TEST_TEXTS = [
    "I am extremely happy and excited about my new opportunity!",
    "I am worried about my exam and I feel nervous about the result.",
    "I am excited about my promotion but nervous about the extra responsibility.",
    "I am furious that they ignored my work and disappointed me again.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./multilabel_emotion_model")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    classifier = MultiLabelBERTClassifier.load(args.model_dir)

    signatures = set()
    all_passed = True

    print("=" * 70)
    print("TASK 1 - EMOTION INTENSITY & EMOTIONAL STATE VALIDATION")
    print("=" * 70)

    for text in TEST_TEXTS:
        state = analyze_emotional_state(
            text, classifier, threshold=args.threshold
        )
        signatures.add(
            (
                state.dominant_emotion,
                tuple(state.multiple_emotions),
                state.emotional_intensity,
                state.polarity,
                state.severity,
            )
        )

        valid_intensity = 0.0 <= state.emotional_intensity <= 1.0
        valid_confidence = 0.0 <= state.emotion_confidence <= 1.0
        valid_probs = all(
            0.0 <= x <= 1.0
            for x in state.emotion_probabilities.values()
        )
        dynamic = state.emotional_intensity not in (0.0, 1.0)

        ok = (
            valid_intensity
            and valid_confidence
            and valid_probs
            and bool(state.multiple_emotions)
            and dynamic
        )
        all_passed = all_passed and ok

        print(f"\nText: {text}")
        print(f"  Dominant emotion : {state.dominant_emotion}")
        print(f"  Multiple emotions: {', '.join(state.multiple_emotions)}")
        print(f"  Confidence       : {state.emotion_confidence:.4f}")
        print(f"  Intensity        : {state.emotional_intensity:.4f}")
        print(f"  Polarity         : {state.polarity}")
        print(f"  Mixed state      : {state.mixed_emotional_state}")
        print(f"  Severity         : {state.severity}")
        print(f"  Validation       : {'PASS' if ok else 'FAIL'}")

    unique_outputs = len(signatures) >= 2
    all_passed = all_passed and unique_outputs

    print("\n" + "-" * 70)
    print(
        "Different inputs produce changing emotional states: "
        f"{'PASS' if unique_outputs else 'FAIL'}"
    )
    print(
        "\nTASK 1 OVERALL: "
        f"{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
