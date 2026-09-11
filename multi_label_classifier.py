"""
Mood Mentor - Milestone 2 Task 3
=================================

TRUE Multi-Label Emotion Classification

The model is trained independently for each emotion using
multi-label BCE loss.

Supported emotions:
    Joy
    Sadness
    Anger
    Fear
    Surprise
    Disgust

The model outputs independent sigmoid probabilities, so
multiple emotions can be detected at the same time.

Example:
    "I am excited but nervous."

Possible result:
    Joy + Fear + Surprise

No prediction or confidence values are hardcoded.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch

from emotion_labels import (
    EMOTION_LABELS,
    merge_training_probs_sigmoid,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_DIR = "./multilabel_emotion_model"

DEFAULT_THRESHOLD = 0.35

MAX_SEQUENCE_LENGTH = 128


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class MultiLabelPrediction:
    text: str

    detected_emotions: List[str]

    primary_emotion: str

    all_scores: Dict[str, float]


# ============================================================
# MULTI-LABEL CLASSIFIER
# ============================================================

class MultiLabelBERTClassifier:

    def __init__(
        self,
        model,
        tokenizer,
        label_list=None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if label_list is None:
            label_list = EMOTION_LABELS

        self.label_list = list(label_list)

    # --------------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------------

    @classmethod
    def load(cls, model_dir=DEFAULT_MODEL_DIR):

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        )

        # Read labels saved with the model if available.
        label_list = None

        if hasattr(model.config, "id2label"):

            id2label = model.config.id2label

            if id2label:

                label_list = [
                    id2label[i]
                    for i in range(len(id2label))
                ]

        if not label_list:
            label_list = EMOTION_LABELS

        # Normalize labels.
        label_list = [
            str(label).strip()
            for label in label_list
        ]

        return cls(
            model=model,
            tokenizer=tokenizer,
            label_list=label_list,
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        text: str,
        threshold: float = DEFAULT_THRESHOLD,
    ):

        if not text or not text.strip():

            raise ValueError(
                "Input text cannot be empty."
            )

        if not 0.0 < threshold < 1.0:

            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.model.eval()

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            logits = outputs.logits[0]

            # IMPORTANT:
            # Multi-label classification uses sigmoid,
            # NOT softmax.
            probabilities = torch.sigmoid(
                logits
            ).cpu().tolist()

        raw_scores = {}

        for index, label in enumerate(
            self.label_list
        ):

            if index < len(probabilities):

                raw_scores[label] = round(
                    float(probabilities[index]),
                    4,
                )

        # Convert model labels into the project's
        # six required emotion categories when needed.
        if set(self.label_list) != set(EMOTION_LABELS):

            all_scores = merge_training_probs_sigmoid(
                raw_scores
            )

        else:

            all_scores = raw_scores

        # ----------------------------------------------------
        # PRIMARY EMOTION
        # ----------------------------------------------------

        primary_emotion = max(
            all_scores.items(),
            key=lambda item: item[1],
        )[0]

        # ----------------------------------------------------
        # MULTI-LABEL DETECTION
        # ----------------------------------------------------

        detected_emotions = [

            emotion

            for emotion, score
            in all_scores.items()

            if score >= threshold
        ]

        # Task 3 requires one or more emotions.
        if not detected_emotions:

            detected_emotions = [
                primary_emotion
            ]

        # Highest score first.
        detected_emotions.sort(
            key=lambda emotion:
                all_scores[emotion],
            reverse=True,
        )

        return MultiLabelPrediction(

            text=text,

            detected_emotions=detected_emotions,

            primary_emotion=primary_emotion,

            all_scores=all_scores,
        )


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION
# ============================================================

def predict_multi_label(
    classifier,
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
):

    """
    Compatibility wrapper.

    Allows Task 4, Task 7 and other modules to call:

        predict_multi_label(
            classifier,
            text,
            threshold
        )

    without directly depending on the class implementation.
    """

    return classifier.predict(
        text,
        threshold=threshold,
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def print_result(
    prediction: MultiLabelPrediction,
    threshold: float,
):

    print("\n" + "=" * 70)

    print(
        " TASK 3 MULTI-LABEL PREDICTION"
    )

    print("=" * 70)

    print("\nInput:")

    print(prediction.text)

    print("\nDetected Emotions:")

    for emotion in prediction.detected_emotions:

        print(
            f"  {emotion}"
        )

    print(
        f"\nPrimary Emotion: "
        f"{prediction.primary_emotion}"
    )

    print(
        f"Threshold: {threshold}"
    )

    print("\nEmotion Scores:")

    for emotion, score in sorted(
        prediction.all_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        bar = "#" * int(
            score * 40
        )

        print(
            f"  {emotion:10s} "
            f"{score:.4f}  "
            f"{bar}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mood Mentor - "
            "Task 3 Multi-Label Emotion Classification"
        )
    )

    parser.add_argument(
        "--text",
        help="Text to analyze",
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to trained multi-label model",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Emotion detection threshold "
            f"(default {DEFAULT_THRESHOLD})"
        ),
    )

    args = parser.parse_args()

    try:

        classifier = (
            MultiLabelBERTClassifier.load(
                args.model_dir
            )
        )

        if args.text:

            prediction = classifier.predict(
                args.text,
                threshold=args.threshold,
            )

            print_result(
                prediction,
                args.threshold,
            )

        else:

            print("=" * 70)

            print(
                " MOOD MENTOR - TASK 3"
            )

            print(
                " TRUE MULTI-LABEL EMOTION CLASSIFICATION"
            )

            print("=" * 70)

            while True:

                text = input(
                    "\nEnter text "
                    "(type 'quit' to exit): "
                ).strip()

                if text.lower() in (
                    "quit",
                    "exit",
                    "q",
                ):

                    print("Bye!")

                    break

                if not text:

                    print(
                        "Please enter some text."
                    )

                    continue

                prediction = (
                    classifier.predict(
                        text,
                        threshold=args.threshold,
                    )
                )

                print_result(
                    prediction,
                    args.threshold,
                )

    except Exception as error:

        print(
            f"\n[ERROR] {error}"
        )


if __name__ == "__main__":
    main()