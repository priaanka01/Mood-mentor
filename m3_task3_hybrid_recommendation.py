"""Mood Mentor - Milestone 3 Task 3: Hybrid Recommendation Engine."""

import argparse
import csv
from typing import List

import numpy as np

from m3_emotion_state import analyze_emotional_state
from multi_label_classifier import MultiLabelBERTClassifier


# =========================================================
# DATA LOADING
# =========================================================

def load_content(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_set(value):
    return {
        x.strip().lower()
        for x in value.split(",")
        if x.strip()
    }


# =========================================================
# 1. RULE-BASED RECOMMENDATION
# =========================================================

def rule_based_score(state, item):
    """
    Applies explicit wellness rules based on
    dominant emotion and emotional intensity.
    """

    emotion = state.dominant_emotion.lower()
    intensity = state.emotional_intensity

    emotions = parse_set(item["emotions"])
    tags = parse_set(item["tags"])

    score = 0.0

    # Emotion rule
    if emotion in emotions:
        score += 0.55

    # Intensity rule
    low = float(item["intensity_min"])
    high = float(item["intensity_max"])

    if low <= intensity <= high:
        score += 0.30

    # High-intensity negative emotion rule
    if intensity >= 0.70:
        if emotion in {"fear", "sadness", "anger"}:
            if tags.intersection(
                {"calm", "stress", "grounding", "support", "regulation"}
            ):
                score += 0.15

    return min(score, 1.0)


# =========================================================
# 2. CONTENT-BASED FILTERING
# =========================================================

def content_based_score(state, item):
    """
    Content-based filtering compares the current
    emotional state with the item's emotion and tags.
    """

    detected = {
        x.lower()
        for x in state.multiple_emotions
    }

    item_emotions = parse_set(item["emotions"])
    item_tags = parse_set(item["tags"])

    if not detected:
        return 0.0

    emotion_overlap = (
        len(detected.intersection(item_emotions))
        / len(detected)
    )

    # Emotion-related tags
    emotion_tags = {
        "calm",
        "stress",
        "grounding",
        "support",
        "connection",
        "reflection",
        "journaling",
        "regulation",
        "mindfulness"
    }

    tag_overlap = len(
        item_tags.intersection(emotion_tags)
    ) / len(emotion_tags)

    return round(
        0.75 * emotion_overlap +
        0.25 * tag_overlap,
        4
    )


# =========================================================
# 3. USER PREFERENCE MATCHING
# =========================================================

PREFERENCE_MAP = {
    "breathing": {"breathing"},
    "mindfulness": {"mindfulness"},
    "walking": {"walk"},
    "walk": {"walk"},
    "social": {"support", "connection"},
    "music": {"music"},
    "journaling": {"journaling"},
    "journal": {"journaling"},
    "grounding": {"grounding"},
    "reflection": {"reflection"},
    "creative": {"creative"},
}


def preference_score(item, preferences):
    if not preferences:
        return 0.0

    tags = parse_set(item["tags"])

    matched = 0

    for preference in preferences:

        preference = preference.lower().strip()

        expected_tags = PREFERENCE_MAP.get(
            preference,
            {preference}
        )

        if tags.intersection(expected_tags):
            matched += 1

    return matched / len(preferences)


# =========================================================
# 4. COLLABORATIVE FILTERING
# =========================================================

# Simulated interaction matrix representing
# previous behavior of multiple users.
#
# 1 = interacted
# 0 = not interacted

USER_INTERACTIONS = {
    "User_A": {
        "W01": 1, "W02": 1, "W03": 0,
        "W04": 0, "W05": 0, "W06": 1,
        "W07": 0, "W08": 1, "W09": 0,
        "W10": 0, "W11": 0, "W12": 0
    },

    "User_B": {
        "W01": 1, "W02": 0, "W03": 0,
        "W04": 0, "W05": 1, "W06": 1,
        "W07": 0, "W08": 1, "W09": 0,
        "W10": 0, "W11": 0, "W12": 0
    },

    "User_C": {
        "W01": 0, "W02": 0, "W03": 1,
        "W04": 0, "W05": 0, "W06": 0,
        "W07": 1, "W08": 0, "W09": 0,
        "W10": 0, "W11": 0, "W12": 1
    },

    "User_D": {
        "W01": 0, "W02": 0, "W03": 0,
        "W04": 1, "W05": 1, "W06": 0,
        "W07": 0, "W08": 0, "W09": 1,
        "W10": 1, "W11": 0, "W12": 0
    }
}


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def collaborative_score(
    candidate_id,
    interactions
):
    """
    Finds users with similar interaction patterns
    and uses their interactions to score unseen content.
    """

    all_ids = list(USER_INTERACTIONS["User_A"].keys())

    current_vector = [
        1 if item in interactions else 0
        for item in all_ids
    ]

    similarities = []

    for user, profile in USER_INTERACTIONS.items():

        user_vector = [
            profile[item]
            for item in all_ids
        ]

        similarity = cosine_similarity(
            current_vector,
            user_vector
        )

        similarities.append(
            (user, similarity)
        )

    weighted_value = 0.0
    total_similarity = 0.0

    for user, similarity in similarities:

        if similarity <= 0:
            continue

        profile = USER_INTERACTIONS[user]

        if candidate_id in profile:
            weighted_value += (
                similarity *
                profile[candidate_id]
            )

            total_similarity += similarity

    if total_similarity == 0:
        return 0.0

    return min(
        weighted_value / total_similarity,
        1.0
    )


# =========================================================
# 5. EMOTION SIMILARITY
# =========================================================

EMOTION_PROFILES = {
    "joy":      {"joy": 1.0},
    "fear":     {"fear": 1.0},
    "sadness":  {"sadness": 1.0},
    "anger":    {"anger": 1.0},
    "disgust":  {"disgust": 1.0},
    "surprise": {"surprise": 1.0}
}


def emotion_similarity(state, item):
    """
    Measures how closely the candidate content emotions
    match the user's detected emotional state.
    """

    detected = {
        x.lower()
        for x in state.multiple_emotions
    }

    item_emotions = parse_set(item["emotions"])

    if not detected or not item_emotions:
        return 0.0

    matches = detected.intersection(item_emotions)

    # More matching emotions = higher similarity
    overlap_score = (
        len(matches) /
        max(len(detected), len(item_emotions))
    )

    # Dominant emotion receives additional importance
    dominant = state.dominant_emotion.lower()

    if dominant in item_emotions:
        overlap_score += 0.35

    return min(overlap_score, 1.0)


# =========================================================
# 6. HISTORICAL USER BEHAVIOR
# =========================================================

def historical_behavior_score(
    item_id,
    history
):
    """
    Encourages unseen content and penalizes content
    already present in recent recommendation history.
    """

    history = [
        x.strip().upper()
        for x in history
    ]

    item_id = item_id.upper()

    if item_id in history:
        # Already recommended -> lower score
        return 0.15

    # Unseen recommendation -> exploration bonus
    return 1.0


# =========================================================
# HYBRID ENGINE
# =========================================================

def calculate_hybrid_score(
    rule_score,
    content_score,
    preference,
    collaborative,
    emotion_sim,
    historical
):

    return min(
        1.0,
        (
            0.20 * rule_score
            + 0.20 * content_score
            + 0.15 * preference
            + 0.15 * collaborative
            + 0.20 * emotion_sim
            + 0.10 * historical
        )
    )


def recommend(
    state,
    content,
    preferences,
    interactions,
    history
):

    results = []

    for item in content:

        rule = rule_based_score(
            state,
            item
        )

        content_score = content_based_score(
            state,
            item
        )

        preference = preference_score(
            item,
            preferences
        )

        collaborative = collaborative_score(
            item["content_id"],
            interactions
        )

        emotion_sim = emotion_similarity(
            state,
            item
        )

        historical = historical_behavior_score(
            item["content_id"],
            history
        )

        hybrid = calculate_hybrid_score(
            rule,
            content_score,
            preference,
            collaborative,
            emotion_sim,
            historical
        )

        row = dict(item)

        row["rule_score"] = round(rule, 4)
        row["content_score"] = round(
            content_score, 4
        )
        row["preference_score"] = round(
            preference, 4
        )
        row["collaborative_score"] = round(
            collaborative, 4
        )
        row["emotion_similarity"] = round(
            emotion_sim, 4
        )
        row["historical_behavior"] = round(
            historical, 4
        )
        row["hybrid_score"] = round(
            hybrid, 4
        )

        results.append(row)

    results.sort(
        key=lambda x: (
            -x["hybrid_score"],
            x["content_id"]
        )
    )

    return results


# =========================================================
# DISPLAY
# =========================================================

def print_results(
    state,
    results,
    preferences,
    interactions,
    history
):

    print("\n" + "=" * 82)
    print("TASK 3 - HYBRID RECOMMENDATION ENGINE")
    print("=" * 82)

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

    print("\nUser profile:")
    print(
        f"  Preferences : "
        f"{', '.join(preferences) or 'None'}"
    )

    print(
        f"  Interactions: "
        f"{', '.join(interactions) or 'None'}"
    )

    print(
        f"  History     : "
        f"{', '.join(history) or 'None'}"
    )

    print("\nRecommendation strategies:")

    print(
        "  Rule-based recommendation       : 20%"
    )

    print(
        "  Content-based filtering          : 20%"
    )

    print(
        "  User preference matching         : 15%"
    )

    print(
        "  Collaborative filtering          : 15%"
    )

    print(
        "  Emotion similarity               : 20%"
    )

    print(
        "  Historical user behavior         : 10%"
    )

    print("\nHybrid recommendation ranking:")

    for index, item in enumerate(
        results[:5],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{item['title']} "
            f"[{item['content_id']}]"
        )

        print(
            f"   HYBRID SCORE       : "
            f"{item['hybrid_score']:.4f}"
        )

        print(
            f"   Rule-based         : "
            f"{item['rule_score']:.4f}"
        )

        print(
            f"   Content-based     : "
            f"{item['content_score']:.4f}"
        )

        print(
            f"   Preference         : "
            f"{item['preference_score']:.4f}"
        )

        print(
            f"   Collaborative     : "
            f"{item['collaborative_score']:.4f}"
        )

        print(
            f"   Emotion similarity: "
            f"{item['emotion_similarity']:.4f}"
        )

        print(
            f"   Historical behavior: "
            f"{item['historical_behavior']:.4f}"
        )

        print(
            f"   {item['description']}"
        )

    print("\n" + "-" * 82)

    print(
        "TASK 3 STATUS: "
        "HYBRID RECOMMENDATIONS GENERATED"
    )

    print("=" * 82)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

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
        "--threshold",
        type=float,
        default=0.35
    )

    args = parser.parse_args()

    classifier = (
        MultiLabelBERTClassifier.load(
            args.model_dir
        )
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

    results = recommend(
        state,
        content,
        preferences,
        interactions,
        history
    )

    print_results(
        state,
        results,
        preferences,
        interactions,
        history
    )