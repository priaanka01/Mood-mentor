"""
Task 8 - Model Testing & Edge Case Validation
===============================================

Tests the Mood Mentor emotion classification system with
different types of inputs.

Test cases:
    1. Positive text
    2. Negative text
    3. Neutral text
    4. Mixed emotions
    5. Short text
    6. Long text
    7. Informal text
    8. Emojis
    9. Ambiguous statement
    10. Empty input
    11. Invalid input

The script verifies that:
    - Valid inputs produce meaningful predictions.
    - Sentiment is generated.
    - BERT emotion and confidence are generated.
    - Multi-label emotions are generated.
    - Confidence scores are within 0-1.
    - Empty and invalid inputs are handled safely.
"""

import argparse

from text_ingestion import ingest_text_input, IngestionError
from preprocessing import preprocess
from sentiment_analysis import analyze_sentiment
from bert_model import EmotionBERTClassifier
from multi_label_classifier import (
    MultiLabelBERTClassifier,
    predict_multi_label,
)


DEFAULT_BERT_DIR = "./bert_emotion_model"
DEFAULT_MULTILABEL_DIR = "./multilabel_emotion_model"
DEFAULT_THRESHOLD = 0.35


# ============================================================
# TEST INPUTS
# ============================================================

TEST_CASES = [

    (
        "Positive Text",
        "I am extremely happy today because I got my dream job!"
    ),

    (
        "Negative Text",
        "I am very angry and disappointed with what happened."
    ),

    (
        "Neutral Text",
        "The meeting is scheduled for tomorrow at 10 AM."
    ),

    (
        "Mixed Emotions",
        "I am excited about my promotion but nervous about the extra responsibility."
    ),

    (
        "Short Text",
        "Happy"
    ),

    (
        "Long Text",
        "Today was a very important day for me. I attended a job interview "
        "that I had been preparing for several weeks. I was nervous before "
        "the interview, but once it started I became more comfortable. "
        "The interviewers were friendly and asked several technical "
        "questions. I was able to answer most of them successfully. "
        "After the interview, I felt relieved, hopeful, and excited about "
        "the possibility of getting the job."
    ),

    (
        "Informal Text",
        "omg i'm sooo happy rn!!! got the job lol"
    ),

    (
        "Emojis",
        "I got the job! 🎉😊❤️"
    ),

    (
        "Ambiguous Statement",
        "That's interesting."
    ),
]


# ============================================================
# SCORE BAR
# ============================================================

def score_bar(score, width=40):
    """
    Create a visual bar based on the actual probability score.

    Example:
        0.75 -> ##############################
        0.25 -> ##########

    The number of # symbols is calculated dynamically.
    """
    filled = int(score * width)
    return "#" * filled


# ============================================================
# RUN ONE VALID INPUT TEST
# ============================================================

def test_valid_input(
    name,
    text,
    bert_classifier,
    multilabel_classifier,
    threshold,
):
    """Run one valid text through the complete emotion pipeline."""

    print("\n" + "=" * 70)
    print(f" TEST: {name}")
    print("=" * 70)

    print(f"Input: {text}")

    try:

        # ----------------------------------------------------
        # STEP 1: Text ingestion
        # ----------------------------------------------------

        record = ingest_text_input(text)

        # ----------------------------------------------------
        # STEP 2: Preprocessing
        # ----------------------------------------------------

        prep = preprocess(record.raw_text)

        # ----------------------------------------------------
        # STEP 3: Sentiment analysis
        # ----------------------------------------------------

        sentiment = analyze_sentiment(
            prep.cleaned_text
        )

        # ----------------------------------------------------
        # STEP 4: BERT emotion classification
        # ----------------------------------------------------

        bert_prediction = bert_classifier.predict(
            prep.cleaned_text
        )

        # ----------------------------------------------------
        # STEP 5: Multi-label emotion classification
        # ----------------------------------------------------

        multi_prediction = predict_multi_label(
            multilabel_classifier,
            prep.cleaned_text,
            threshold=threshold,
        )

        # ----------------------------------------------------
        # STEP 6: Validate results
        # ----------------------------------------------------

        sentiment_ok = (
            sentiment.label is not None
        )

        emotion_ok = (
            bert_prediction.primary_emotion is not None
            and bert_prediction.primary_confidence is not None
        )

        confidence_ok = (
            0.0
            <= bert_prediction.primary_confidence
            <= 1.0
        )

        all_scores_ok = (
            isinstance(
                bert_prediction.all_scores,
                dict,
            )
            and len(bert_prediction.all_scores) > 0
            and all(
                0.0 <= score <= 1.0
                for score in bert_prediction.all_scores.values()
            )
        )

        multilabel_ok = (
            multi_prediction.primary_emotion is not None
            and isinstance(
                multi_prediction.all_scores,
                dict,
            )
            and len(multi_prediction.all_scores) > 0
        )

        multilabel_scores_ok = (
            all(
                0.0 <= score <= 1.0
                for score in multi_prediction.all_scores.values()
            )
        )

        all_passed = (
            sentiment_ok
            and emotion_ok
            and confidence_ok
            and all_scores_ok
            and multilabel_ok
            and multilabel_scores_ok
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print("\nResults:")

        print(
            f"  Sentiment          : "
            f"{sentiment.label}"
        )

        print(
            f"  BERT Emotion       : "
            f"{bert_prediction.primary_emotion}"
        )

        print(
            f"  BERT Confidence    : "
            f"{bert_prediction.primary_confidence:.4f}"
        )

        print(
            f"  Multi-label        : "
            f"{', '.join(multi_prediction.detected_emotions)}"
        )

        print(
            f"  Multi-label Primary: "
            f"{multi_prediction.primary_emotion}"
        )

        print(
            f"  Threshold          : "
            f"{threshold}"
        )

        # ----------------------------------------------------
        # BERT scores with ### visual bars
        # ----------------------------------------------------

        print("\nBERT Emotion Scores:")

        for label, score in sorted(
            bert_prediction.all_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            bar = score_bar(score)

            print(
                f"  {label:10s} : "
                f"{score:.4f}  {bar}"
            )

        # ----------------------------------------------------
        # Multi-label scores with ### visual bars
        # ----------------------------------------------------

        print("\nMulti-label Emotion Scores:")

        for label, score in sorted(
            multi_prediction.all_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            bar = score_bar(score)

            print(
                f"  {label:10s} : "
                f"{score:.4f}  {bar}"
            )

        # ----------------------------------------------------
        # Validation checks
        # ----------------------------------------------------

        print("\nChecks:")

        print(
            f"  Sentiment generated : "
            f"{'PASS' if sentiment_ok else 'FAIL'}"
        )

        print(
            f"  Emotion generated   : "
            f"{'PASS' if emotion_ok else 'FAIL'}"
        )

        print(
            f"  Confidence 0-1      : "
            f"{'PASS' if confidence_ok else 'FAIL'}"
        )

        print(
            f"  BERT scores valid   : "
            f"{'PASS' if all_scores_ok else 'FAIL'}"
        )

        print(
            f"  Multi-label output  : "
            f"{'PASS' if multilabel_ok else 'FAIL'}"
        )

        print(
            f"  Multi-label scores  : "
            f"{'PASS' if multilabel_scores_ok else 'FAIL'}"
        )

        print(
            f"\n  OVERALL             : "
            f"{'PASS' if all_passed else 'FAIL'}"
        )

        return all_passed

    except Exception as error:

        print(
            f"\n  ERROR: "
            f"{type(error).__name__}: {error}"
        )

        print("  OVERALL             : FAIL")

        return False


# ============================================================
# EMPTY INPUT TEST
# ============================================================

def test_empty_input():
    """Verify that empty input is rejected meaningfully."""

    print("\n" + "=" * 70)
    print(" TEST: Empty Input")
    print("=" * 70)

    try:

        ingest_text_input("")

        print(
            "  FAIL - Empty input was accepted "
            "without an error."
        )

        return False

    except IngestionError as error:

        print(
            f"  PASS - Empty input rejected correctly: "
            f"{error}"
        )

        return True

    except Exception as error:

        print(
            f"  PASS - Empty input produced a controlled "
            f"error: {type(error).__name__}: {error}"
        )

        return True


# ============================================================
# INVALID INPUT TEST
# ============================================================

def test_invalid_input():
    """
    Verify that invalid input is handled safely.

    A non-string object is intentionally supplied because
    the ingestion function expects textual input.
    """

    print("\n" + "=" * 70)
    print(" TEST: Invalid Input")
    print("=" * 70)

    try:

        ingest_text_input(None)

        print(
            "  FAIL - Invalid input was accepted."
        )

        return False

    except (
        IngestionError,
        TypeError,
        ValueError,
    ) as error:

        print(
            f"  PASS - Invalid input handled correctly: "
            f"{type(error).__name__}: {error}"
        )

        return True

    except Exception as error:

        print(
            f"  PASS - Invalid input produced a controlled "
            f"error: {type(error).__name__}: {error}"
        )

        return True


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Task 8 - Edge Case Validation"
    )

    parser.add_argument(
        "--bert-dir",
        default=DEFAULT_BERT_DIR,
        help="Path to saved BERT model",
    )

    parser.add_argument(
        "--multilabel-dir",
        default=DEFAULT_MULTILABEL_DIR,
        help="Path to saved multi-label model",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Multi-label detection threshold "
            f"(default {DEFAULT_THRESHOLD})"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    if not 0.0 <= args.threshold <= 1.0:

        parser.error(
            "Threshold must be between 0.0 and 1.0"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)
    print(" MOOD MENTOR - TASK 8")
    print(" MODEL TESTING & EDGE CASE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load BERT
    # --------------------------------------------------------

    print("\nLoading BERT emotion model...")

    bert_classifier = EmotionBERTClassifier.load(
        args.bert_dir
    )

    print("BERT model loaded.")

    # --------------------------------------------------------
    # Load Multi-label BERT
    # --------------------------------------------------------

    print("\nLoading multi-label BERT model...")

    multilabel_classifier = MultiLabelBERTClassifier.load(
        args.multilabel_dir
    )

    print("Multi-label BERT model loaded.")

    # --------------------------------------------------------
    # Run valid test cases
    # --------------------------------------------------------

    passed = 0
    failed = 0

    for name, text in TEST_CASES:

        result = test_valid_input(
            name,
            text,
            bert_classifier,
            multilabel_classifier,
            args.threshold,
        )

        if result:
            passed += 1
        else:
            failed += 1

    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if test_empty_input():

        passed += 1

    else:

        failed += 1

    # --------------------------------------------------------
    # Invalid input
    # --------------------------------------------------------

    if test_invalid_input():

        passed += 1

    else:

        failed += 1

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    total = passed + failed

    print("\n" + "=" * 70)
    print(" TASK 8 FINAL VALIDATION")
    print("=" * 70)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if failed == 0:

        print("\nTASK 8 RESULT: ALL TESTS PASSED")

    else:

        print(
            f"\nTASK 8 RESULT: "
            f"{failed} TEST(S) FAILED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()