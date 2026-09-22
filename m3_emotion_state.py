"""
Mood Mentor - Milestone 3 Task 1
Emotion Intensity & Emotional State Analysis

Uses the existing Milestone 2 multi-label BERT model and Milestone 1
VADER sentiment model. No emotion probabilities are hardcoded.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import json

from sentiment_analysis import analyze_sentiment
from multi_label_classifier import MultiLabelBERTClassifier, predict_multi_label


EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]


@dataclass
class EmotionalState:
    text: str
    dominant_emotion: str
    multiple_emotions: List[str]
    emotion_confidence: float
    emotional_intensity: float
    polarity: str
    polarity_score: float
    mixed_emotional_state: bool
    severity: str
    emotion_probabilities: Dict[str, float]

    def as_dict(self):
        return asdict(self)


def calculate_intensity(scores: Dict[str, float]) -> float:
    """Dynamic intensity from the model's emotion probabilities."""
    values = sorted(scores.values(), reverse=True)
    if not values:
        return 0.0

    strongest = values[0]
    supporting = sum(values[1:3]) / max(1, len(values[1:3]))

    # Strong primary emotion + supporting emotions = higher intensity.
    intensity = (0.70 * strongest) + (0.30 * supporting)
    return round(max(0.0, min(1.0, intensity)), 4)


def classify_severity(intensity: float, polarity: str, negative_score: float) -> str:
    """Severity is a wellness-routing signal, not a medical diagnosis."""
    if polarity == "Negative":
        if intensity >= 0.75 or negative_score >= 0.75:
            return "High"
        if intensity >= 0.45 or negative_score >= 0.45:
            return "Moderate"
        return "Low"

    if intensity >= 0.80:
        return "High"
    if intensity >= 0.50:
        return "Moderate"
    return "Low"


def analyze_emotional_state(
    text: str,
    classifier: MultiLabelBERTClassifier,
    threshold: float = 0.35,
) -> EmotionalState:
    sentiment = analyze_sentiment(text)
    prediction = predict_multi_label(classifier, text, threshold=threshold)

    scores = prediction.all_scores
    intensity = calculate_intensity(scores)

    detected = prediction.detected_emotions
    top_values = sorted(scores.values(), reverse=True)
    mixed = len(detected) >= 2 and (
        len(top_values) >= 2 and (top_values[0] - top_values[1]) <= 0.25
    )

    severity = classify_severity(
        intensity=intensity,
        polarity=sentiment.label,
        negative_score=sentiment.neg,
    )

    return EmotionalState(
        text=text,
        dominant_emotion=prediction.primary_emotion,
        multiple_emotions=detected,
        emotion_confidence=round(
            scores[prediction.primary_emotion], 4
        ),
        emotional_intensity=intensity,
        polarity=sentiment.label,
        polarity_score=round(sentiment.compound, 4),
        mixed_emotional_state=mixed,
        severity=severity,
        emotion_probabilities={
            k: round(float(v), 4) for k, v in scores.items()
        },
    )


def print_state(state: EmotionalState):
    print("\n" + "=" * 70)
    print("MOOD MENTOR - TASK 1: EMOTIONAL STATE ANALYSIS")
    print("=" * 70)
    print(f"Text                 : {state.text}")
    print(f"Dominant emotion     : {state.dominant_emotion}")
    print(f"Multiple emotions    : {', '.join(state.multiple_emotions)}")
    print(f"Emotion confidence   : {state.emotion_confidence:.4f}")
    print(f"Emotional intensity  : {state.emotional_intensity:.4f}")
    print(f"Polarity             : {state.polarity}")
    print(f"Polarity score       : {state.polarity_score:.4f}")
    print(f"Mixed emotional state: {'YES' if state.mixed_emotional_state else 'NO'}")
    print(f"Emotion severity     : {state.severity}")
    print("\nEmotion probabilities:")
    for emotion, score in sorted(
        state.emotion_probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {emotion:10s}: {score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", default="./multilabel_emotion_model")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    classifier = MultiLabelBERTClassifier.load(args.model_dir)
    state = analyze_emotional_state(
        args.text, classifier, threshold=args.threshold
    )
    print_state(state)
