"""Mood Mentor - Milestone 3 Task 2: Personalized Recommendation Model."""

import argparse
import csv
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor


FEATURE_NAMES = [
    "emotion_relevance",
    "intensity_relevance",
    "preference_match",
    "previous_interaction",
    "history_penalty",
    "emotional_trend",
]


# ---------------------------------------------------------
# Load wellness content
# ---------------------------------------------------------

def load_content(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def normalize(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def parse_set(value: str) -> set:
    return {
        normalize(x)
        for x in value.split(",")
        if x.strip()
    }


# ---------------------------------------------------------
# Preference aliases
# ---------------------------------------------------------

PREFERENCE_ALIASES = {
    "breathing": {"breathing"},
    "mindfulness": {"mindfulness"},
    "walking": {"walk", "walking"},
    "walk": {"walk", "walking"},
    "social": {"support", "connection"},
    "grounding": {"grounding"},
    "journaling": {"journaling"},
    "journal": {"journaling"},
    "music": {"music"},
    "reflection": {"reflection"},
    "creative": {"creative"},
    "self compassion": {"self-compassion"},
}


# ---------------------------------------------------------
# Feature 1: Emotion relevance
# ---------------------------------------------------------

def emotion_match(state, item):
    content_emotions = parse_set(item["emotions"])

    dominant = normalize(state.dominant_emotion)

    multiple = {
        normalize(x)
        for x in state.multiple_emotions
    }

    # Stronger importance for dominant emotion
    if dominant in content_emotions:
        score = 1.0
    else:
        matched = len(content_emotions.intersection(multiple))

        if matched == 0:
            score = 0.0
        else:
            score = min(0.6, 0.3 * matched)

    return score


# ---------------------------------------------------------
# Feature 2: Emotional intensity relevance
# ---------------------------------------------------------

def intensity_match(state, item):
    low = float(item["intensity_min"])
    high = float(item["intensity_max"])

    intensity = state.emotional_intensity

    if low <= intensity <= high:
        return 1.0

    if intensity < low:
        distance = low - intensity
    else:
        distance = intensity - high

    return max(0.0, 1.0 - distance)


# ---------------------------------------------------------
# Feature 3: User preference
# ---------------------------------------------------------

def preference_match(item, preferences):
    if not preferences:
        return 0.0

    tags = parse_set(item["tags"])

    matched = 0

    for preference in preferences:
        preference = normalize(preference)

        aliases = PREFERENCE_ALIASES.get(
            preference,
            {preference}
        )

        if tags.intersection(aliases):
            matched += 1

    return matched / len(preferences)


# ---------------------------------------------------------
# Feature 4: Previous interaction
# ---------------------------------------------------------

def interaction_match(item, interactions):
    ids = {
        x.strip().upper()
        for x in interactions
    }

    return (
        1.0
        if item["content_id"].upper() in ids
        else 0.0
    )


# ---------------------------------------------------------
# Feature 5: Recommendation history penalty
# ---------------------------------------------------------

def history_penalty(item, history):
    ids = {
        x.strip().upper()
        for x in history
    }

    return (
        1.0
        if item["content_id"].upper() in ids
        else 0.0
    )


# ---------------------------------------------------------
# Feature 6: Emotional trend
# ---------------------------------------------------------

def trend_match(state, trend):

    if not trend:
        return 0.0

    recent = {
        normalize(x)
        for x in trend[-3:]
    }

    detected = {
        normalize(x)
        for x in state.multiple_emotions
    }

    return (
        1.0
        if detected.intersection(recent)
        else 0.0
    )


# ---------------------------------------------------------
# Build feature vector
# ---------------------------------------------------------

def build_features(
    state,
    item,
    preferences,
    interactions,
    history,
    trend,
):

    return np.array([
        emotion_match(state, item),
        intensity_match(state, item),
        preference_match(item, preferences),
        interaction_match(item, interactions),
        history_penalty(item, history),
        trend_match(state, trend),
    ], dtype=float)


# ---------------------------------------------------------
# Training data for Random Forest
# ---------------------------------------------------------

def training_data():

    X = np.array([

        # emotion intensity preference interaction history trend

        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.9, 0.5, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.6, 1.0, 1.0, 1.0, 0.0, 1.0],
        [0.6, 1.0, 0.5, 1.0, 0.0, 1.0],

        [0.6, 0.9, 0.0, 0.0, 0.0, 1.0],
        [0.3, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.3, 0.8, 0.5, 0.0, 0.0, 1.0],

        [0.0, 0.7, 1.0, 0.0, 0.0, 0.0],
        [0.3, 0.6, 0.0, 0.0, 0.0, 0.0],

        # Previously recommended items
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
        [0.6, 0.9, 1.0, 0.0, 1.0, 1.0],
        [0.3, 0.8, 0.0, 0.0, 1.0, 1.0],

    ], dtype=float)

    y = np.array([
        0.98,
        0.94,
        0.86,
        0.90,
        0.91,
        0.88,
        0.76,
        0.82,
        0.72,
        0.55,
        0.45,
        0.35,
        0.30,
        0.28,
        0.20,
    ])

    return X, y


# ---------------------------------------------------------
# Train ML model
# ---------------------------------------------------------

def train_model():

    X, y = training_data()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    model.fit(X, y)

    return model


# ---------------------------------------------------------
# Personalized recommendation engine
# ---------------------------------------------------------

def recommend(
    state,
    content,
    preferences=None,
    interactions=None,
    history=None,
    emotional_trend=None,
    top_k=5,
):

    preferences = preferences or []
    interactions = interactions or []
    history = history or []
    emotional_trend = emotional_trend or []

    model = train_model()

    results = []

    for item in content:

        features = build_features(
            state,
            item,
            preferences,
            interactions,
            history,
            emotional_trend,
        )

        ml_score = float(
            model.predict([features])[0]
        )

        emotion_score = features[0]
        intensity_score = features[1]
        preference_score = features[2]
        interaction_score = features[3]
        history_score = features[4]
        trend_score = features[5]

        # -------------------------------------------------
        # Explicit personalization layer
        # -------------------------------------------------

        final_score = (

            0.30 * emotion_score
            + 0.20 * intensity_score
            + 0.20 * preference_score
            + 0.15 * interaction_score
            + 0.10 * trend_score
            + 0.05 * ml_score

        )

        # Recommendation history penalty
        if history_score == 1.0:
            final_score -= 0.15

        final_score = max(
            0.0,
            min(1.0, final_score)
        )

        row = dict(item)

        row["personalized_score"] = round(
            final_score,
            4
        )

        row["ml_score"] = round(
            ml_score,
            4
        )

        row["feature_vector"] = features.tolist()

        results.append(row)

    # Dynamic ranking
    results.sort(
        key=lambda x: (
            -x["personalized_score"],
            -x["ml_score"],
            x["content_id"]
        )
    )

    return results[:top_k]


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

def print_recommendations(
    state,
    results,
    preferences,
    interactions,
    history,
    trend,
):

    print("\n" + "=" * 78)
    print("TASK 2 - PERSONALIZED RECOMMENDATION MODEL")
    print("=" * 78)

    print("\nUser emotional state:")

    print(
        f"  Dominant emotion : "
        f"{state.dominant_emotion}"
    )

    print(
        f"  Multiple emotions: "
        f"{', '.join(state.multiple_emotions)}"
    )

    print(
        f"  Intensity        : "
        f"{state.emotional_intensity:.4f}"
    )

    print(
        f"  Severity         : "
        f"{state.severity}"
    )

    print("\nPersonalization inputs:")

    print(
        f"  User preferences       : "
        f"{', '.join(preferences) or 'None'}"
    )

    print(
        f"  Previous interactions  : "
        f"{', '.join(interactions) or 'None'}"
    )

    print(
        f"  Recommendation history : "
        f"{', '.join(history) or 'None'}"
    )

    print(
        f"  Emotional trends       : "
        f"{', '.join(trend) or 'None'}"
    )

    print("\nML model:")
    print("  Random Forest Regressor")

    print("\nPersonalization formula:")
    print(
        "  Emotion 30% + Intensity 20% + "
        "Preference 20% + Interaction 15% + "
        "Trend 10% + ML 5%"
    )

    print("\nFeatures used:")

    for i, name in enumerate(
        FEATURE_NAMES,
        1
    ):
        print(
            f"  {i}. "
            f"{name.replace('_', ' ').title()}"
        )

    print("\nTop personalized recommendations:")

    for i, item in enumerate(
        results,
        1
    ):

        f = item["feature_vector"]

        print(
            f"\n{i}. {item['title']} "
            f"[{item['content_id']}]"
        )

        print(
            f"   Personalized score : "
            f"{item['personalized_score']:.4f}"
        )

        print(
            f"   ML score           : "
            f"{item['ml_score']:.4f}"
        )

        print(
            f"   {item['description']}"
        )

        print(
            f"   Emotion relevance  : "
            f"{f[0]:.2f}"
        )

        print(
            f"   Intensity relevance: "
            f"{f[1]:.2f}"
        )

        print(
            f"   Preference match   : "
            f"{f[2]:.2f}"
        )

        print(
            f"   Previous interaction: "
            f"{f[3]:.2f}"
        )

        print(
            f"   History penalty    : "
            f"{f[4]:.2f}"
        )

        print(
            f"   Emotional trend    : "
            f"{f[5]:.2f}"
        )

    print("\n" + "-" * 78)

    print(
        "TASK 2 STATUS: "
        "PERSONALIZED RECOMMENDATIONS GENERATED"
    )

    print("=" * 78)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    from multi_label_classifier import (
        MultiLabelBERTClassifier
    )

    from m3_emotion_state import (
        analyze_emotional_state
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        required=True
    )

    parser.add_argument(
        "--model-dir",
        default="./multilabel_emotion_model"
    )

    parser.add_argument(
        "--content",
        default="wellness_content.csv"
    )

    parser.add_argument(
        "--preferences",
        default="breathing,mindfulness"
    )

    parser.add_argument(
        "--interactions",
        default="W06,W08"
    )

    parser.add_argument(
        "--history",
        default="W03,W07"
    )

    parser.add_argument(
        "--trend",
        default="Fear,Fear,Sadness"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    classifier = MultiLabelBERTClassifier.load(
        args.model_dir
    )

    state = analyze_emotional_state(
        args.text,
        classifier,
        threshold=args.threshold
    )

    content = load_content(
        args.content
    )

    preferences = [
        x.strip()
        for x in args.preferences.split(",")
        if x.strip()
    ]

    interactions = [
        x.strip()
        for x in args.interactions.split(",")
        if x.strip()
    ]

    history = [
        x.strip()
        for x in args.history.split(",")
        if x.strip()
    ]

    trend = [
        x.strip()
        for x in args.trend.split(",")
        if x.strip()
    ]

    results = recommend(
        state,
        content,
        preferences,
        interactions,
        history,
        trend,
        args.top_k
    )

    print_recommendations(
        state,
        results,
        preferences,
        interactions,
        history,
        trend
    )