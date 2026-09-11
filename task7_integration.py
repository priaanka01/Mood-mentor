"""
Mood Mentor - Milestone 2 Task 7
=================================

Milestone 1 + Milestone 2 Integration

Complete workflow:

    Text Input
        ↓
    Text Ingestion / Validation
        ↓
    Preprocessing
        ↓
    VADER Sentiment Analysis
        ↓
    BERT Emotion Classification
        ↓
    Multi-Label Emotion Detection
        ↓
    Confidence / Consistency Check
        ↓
    Final Combined Analysis

Important:
- BERT and Multi-Label BERT are separate models.
- Sentiment and emotion are separate outputs.
- No sentiment, emotion, or confidence values are hardcoded.
"""

import argparse
import json
import sys

from text_ingestion import (
    ingest_text_input,
    ingest_txt_file,
    IngestionError,
)

from preprocessing import preprocess
from sentiment_analysis import analyze_sentiment

from bert_model import EmotionBERTClassifier

from multi_label_classifier import (
    MultiLabelBERTClassifier,
    predict_multi_label,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BERT_DIR = "./bert_emotion_model"

DEFAULT_MULTILABEL_DIR = "./multilabel_emotion_model"

DEFAULT_THRESHOLD = 0.35

MIN_PRIMARY_CONFIDENCE = 0.40

EMOTION_MARGIN = 0.08


# ============================================================
# FINAL EMOTION DECISION
# ============================================================

def determine_final_emotion(bert_prediction):
    """
    Determine final emotion using real BERT probabilities.

    No emotion values are hardcoded.
    """

    scores = bert_prediction.all_scores

    if not scores:
        return {
            "emotion": "Unknown",
            "confidence": 0.0,
            "status": "No emotion scores available",
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_emotion, top_score = ranked[0]

    if len(ranked) > 1:
        second_emotion, second_score = ranked[1]
    else:
        second_emotion, second_score = None, 0.0

    margin = top_score - second_score

    # Strong prediction
    if (
        top_score >= MIN_PRIMARY_CONFIDENCE
        and margin >= EMOTION_MARGIN
    ):
        return {
            "emotion": top_emotion,
            "confidence": top_score,
            "status": "Confident",
        }

    # Close competition
    if margin < EMOTION_MARGIN:
        return {
            "emotion": f"Mixed ({top_emotion} + {second_emotion})",
            "confidence": top_score,
            "status": "Mixed/Uncertain",
        }

    # Low confidence
    return {
        "emotion": top_emotion,
        "confidence": top_score,
        "status": "Low Confidence",
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_text(
    text: str,
    bert_dir: str = DEFAULT_BERT_DIR,
    multilabel_dir: str = DEFAULT_MULTILABEL_DIR,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Run complete Milestone 1 + Milestone 2 Task 7 pipeline.

    Two different models are used:

    1. BERT model
       ./bert_emotion_model

    2. Multi-label BERT model
       ./multilabel_emotion_model
    """

    # --------------------------------------------------------
    # STEP 1: TEXT INGESTION
    # --------------------------------------------------------

    record = ingest_text_input(text)

    # --------------------------------------------------------
    # STEP 2: PREPROCESSING
    # --------------------------------------------------------

    prep = preprocess(record.raw_text)

    # --------------------------------------------------------
    # STEP 3: SENTIMENT ANALYSIS
    # --------------------------------------------------------

    sentiment = analyze_sentiment(
        prep.cleaned_text
    )

    # --------------------------------------------------------
    # STEP 4: LOAD SINGLE-LABEL BERT
    # --------------------------------------------------------

    print("\nLoading BERT emotion model...")

    bert_classifier = EmotionBERTClassifier.load(
        bert_dir
    )

    # --------------------------------------------------------
    # STEP 5: SINGLE-LABEL BERT PREDICTION
    # --------------------------------------------------------

    bert_prediction = bert_classifier.predict(
        prep.cleaned_text
    )

    # --------------------------------------------------------
    # STEP 6: LOAD MULTI-LABEL BERT
    # --------------------------------------------------------

    print("\nLoading multi-label BERT model...")

    multilabel_classifier = MultiLabelBERTClassifier.load(
        multilabel_dir
    )

    # --------------------------------------------------------
    # STEP 7: MULTI-LABEL PREDICTION
    # --------------------------------------------------------

    multi_prediction = predict_multi_label(
        multilabel_classifier,
        prep.cleaned_text,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # STEP 8: FINAL EMOTION
    # --------------------------------------------------------

    final_emotion = determine_final_emotion(
        bert_prediction
    )

    # --------------------------------------------------------
    # STEP 9: BUILD RESULT
    # --------------------------------------------------------

    result = {
        "input_text": prep.original_text,

        "processed_text": prep.processed_text,

        "sentiment": {
            "label": sentiment.label,
            "compound": round(
                sentiment.compound,
                4
            ),
            "positive": round(
                sentiment.pos,
                4
            ),
            "negative": round(
                sentiment.neg,
                4
            ),
            "neutral": round(
                sentiment.neu,
                4
            ),
        },

        # ----------------------------------------------------
        # SINGLE-LABEL BERT
        # ----------------------------------------------------

        "emotion": {
            "primary_emotion":
                bert_prediction.primary_emotion,

            "primary_confidence":
                round(
                    bert_prediction.primary_confidence,
                    4
                ),

            "all_scores": {
                emotion: round(
                    score,
                    4
                )
                for emotion, score
                in bert_prediction.all_scores.items()
            },
        },

        # ----------------------------------------------------
        # FINAL EMOTION
        # ----------------------------------------------------

        "final_emotion": {
            "emotion":
                final_emotion["emotion"],

            "confidence":
                round(
                    final_emotion["confidence"],
                    4
                ),

            "status":
                final_emotion["status"],
        },

        # ----------------------------------------------------
        # MULTI-LABEL MODEL
        # ----------------------------------------------------

        "multi_label": {
            "detected_emotions":
                multi_prediction.detected_emotions,

            "primary_emotion":
                multi_prediction.primary_emotion,

            "all_scores": {
                emotion: round(
                    score,
                    4
                )
                for emotion, score
                in multi_prediction.all_scores.items()
            },

            "threshold":
                threshold,
        },
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 70)
    print(" MOOD MENTOR - TASK 7 INTEGRATED ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    print("\n[1] INPUT TEXT")
    print("-" * 70)

    print(
        result["input_text"]
    )

    # --------------------------------------------------------
    # PROCESSED TEXT
    # --------------------------------------------------------

    print("\n[2] PROCESSED TEXT")
    print("-" * 70)

    print(
        result["processed_text"]
    )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    print("\n[3] VADER SENTIMENT")
    print("-" * 70)

    sentiment = result["sentiment"]

    print(
        f"Sentiment      : "
        f"{sentiment['label']}"
    )

    print(
        f"Compound Score : "
        f"{sentiment['compound']}"
    )

    print(
        f"Positive Score : "
        f"{sentiment['positive']}"
    )

    print(
        f"Negative Score : "
        f"{sentiment['negative']}"
    )

    print(
        f"Neutral Score  : "
        f"{sentiment['neutral']}"
    )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    print("\n[4] BERT EMOTION")
    print("-" * 70)

    emotion = result["emotion"]

    print(
        f"Primary Emotion : "
        f"{emotion['primary_emotion']}"
    )

    print(
        f"Confidence      : "
        f"{emotion['primary_confidence']}"
    )

    print("\nAll BERT Emotion Scores:")

    for label, score in sorted(
        emotion["all_scores"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        bar = "#" * int(
            score * 40
        )

        print(
            f"  {label:10s} "
            f"{score:.4f}  "
            f"{bar}"
        )

    # --------------------------------------------------------
    # MULTI-LABEL
    # --------------------------------------------------------

    print("\n[5] MULTI-LABEL EMOTION")
    print("-" * 70)

    multi = result["multi_label"]

    print(
        "Detected Emotions : "
        + ", ".join(
            multi["detected_emotions"]
        )
    )

    print(
        f"Primary Emotion   : "
        f"{multi['primary_emotion']}"
    )

    print(
        f"Threshold         : "
        f"{multi['threshold']}"
    )

    print(
        "\nIndependent Emotion Scores:"
    )

    for label, score in sorted(
        multi["all_scores"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        bar = "#" * int(
            score * 40
        )

        print(
            f"  {label:10s} "
            f"{score:.4f}  "
            f"{bar}"
        )

    # --------------------------------------------------------
    # FINAL EMOTION
    # --------------------------------------------------------

    final_emotion = result["final_emotion"]

    print(
        "\n[6] EMOTION CONSISTENCY CHECK"
    )

    print("-" * 70)

    print(
        f"Final Emotion : "
        f"{final_emotion['emotion']}"
    )

    print(
        f"Confidence    : "
        f"{final_emotion['confidence']}"
    )

    print(
        f"Status        : "
        f"{final_emotion['status']}"
    )

    # --------------------------------------------------------
    # FINAL ANALYSIS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        " FINAL ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Sentiment : "
        f"{sentiment['label']}"
    )

    print(
        f"Emotion   : "
        f"{final_emotion['emotion']}"
    )

    print(
        f"Confidence: "
        f"{final_emotion['confidence']}"
    )

    print(
        "Detected  : "
        + ", ".join(
            multi["detected_emotions"]
        )
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
            "Mood Mentor - Milestone 2 Task 7 "
            "Integration"
        )
    )

    parser.add_argument(
        "--text",
        help="Direct text input",
    )

    parser.add_argument(
        "--txt",
        help="Path to a .txt file",
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_BERT_DIR,
        help="Path to saved BERT model",
    )

    parser.add_argument(
        "--multilabel-dir",
        default=DEFAULT_MULTILABEL_DIR,
        help="Path to saved multi-label BERT model",
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

    parser.add_argument(
        "--json-out",
        help="Optional path to save JSON result",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DETERMINE INPUT
    # --------------------------------------------------------

    try:

        if args.text:

            record = ingest_text_input(
                args.text
            )

        elif args.txt:

            record = ingest_txt_file(
                args.txt
            )

        else:

            print(
                "=" * 70
            )

            print(
                " MOOD MENTOR - TASK 7"
            )

            print(
                " Milestone 1 + Milestone 2 Integration"
            )

            print(
                "=" * 70
            )

            text = input(
                "\nEnter text to analyze: "
            ).strip()

            record = ingest_text_input(
                text
            )

        # ----------------------------------------------------
        # RUN PIPELINE
        # ----------------------------------------------------

        result = analyze_text(
            record.raw_text,

            bert_dir=args.model_dir,

            multilabel_dir=args.multilabel_dir,

            threshold=args.threshold,
        )

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print_result(
            result
        )

        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        if args.json_out:

            with open(
                args.json_out,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"\nJSON result saved to: "
                f"{args.json_out}"
            )

    except IngestionError as error:

        print(
            f"\n[INPUT ERROR] {error}"
        )

        sys.exit(1)

    except FileNotFoundError as error:

        print(
            f"\n[FILE ERROR] {error}"
        )

        sys.exit(1)

    except Exception as error:

        print(
            f"\n[ERROR] Task 7 pipeline failed: "
            f"{error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()