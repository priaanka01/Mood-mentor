import argparse
from collections import Counter


# ---------------------------------------------------------
# WELLNESS CONTENT
# ---------------------------------------------------------

WELLNESS_CONTENT = [
    {
        "id": "W01",
        "title": "Deep Breathing",
        "emotions": ["Fear", "Anxiety"],
        "tags": ["breathing", "calm", "stress"]
    },
    {
        "id": "W02",
        "title": "Grounding 5-4-3-2-3",
        "emotions": ["Fear", "Sadness"],
        "tags": ["grounding", "stress", "calm"]
    },
    {
        "id": "W03",
        "title": "Positive Reflection",
        "emotions": ["Joy", "Surprise"],
        "tags": ["gratitude", "positive", "motivation"]
    },
    {
        "id": "W04",
        "title": "Healthy Anger Release",
        "emotions": ["Anger", "Disgust"],
        "tags": ["anger", "journaling", "regulation"]
    },
    {
        "id": "W05",
        "title": "Calm Music",
        "emotions": ["Fear", "Sadness", "Anger"],
        "tags": ["music", "relaxation", "calm"]
    },
    {
        "id": "W06",
        "title": "Mindful Walk",
        "emotions": ["Sadness", "Fear", "Anger"],
        "tags": ["mindfulness", "walk", "stress"]
    },
    {
        "id": "W07",
        "title": "Achievement Journal",
        "emotions": ["Joy", "Surprise"],
        "tags": ["confidence", "reflection", "positive"]
    },
    {
        "id": "W08",
        "title": "Self Compassion",
        "emotions": ["Sadness", "Fear"],
        "tags": ["self-compassion", "reflection"]
    },
    {
        "id": "W09",
        "title": "Focus Reset",
        "emotions": ["Fear", "Anger", "Sadness"],
        "tags": ["focus", "study", "stress"]
    },
    {
        "id": "W10",
        "title": "Creative Expression",
        "emotions": ["Sadness", "Anger", "Disgust"],
        "tags": ["creative", "journaling", "expression"]
    },
    {
        "id": "W11",
        "title": "Social Connection",
        "emotions": ["Sadness", "Fear"],
        "tags": ["support", "connection"]
    },
    {
        "id": "W12",
        "title": "Celebration Playlist",
        "emotions": ["Joy", "Surprise"],
        "tags": ["music", "energy", "motivation"]
    }
]


# ---------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------

def parse_list(value):
    if not value:
        return []

    return [
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    ]


# ---------------------------------------------------------
# RECOMMENDATION FEATURE CALCULATION
# ---------------------------------------------------------

def calculate_features(
    item,
    emotion,
    intensity,
    preferences,
    interactions,
    history,
    trend
):
    emotion_score = 0.0

    if emotion in item["emotions"]:
        emotion_score = 1.0

    preference_score = 0.0

    for preference in preferences:
        if (
            preference in item["tags"]
            or preference in item["title"].lower()
        ):
            preference_score = 1.0
            break

    interaction_score = (
        1.0
        if item["id"] in interactions
        else 0.0
    )

    history_score = (
        0.0
        if item["id"] in history
        else 1.0
    )

    trend_score = 0.0

    if trend in item["emotions"]:
        trend_score = 1.0

    # Higher intensity gives more importance
    # to emotion-related content.
    intensity_score = (
        intensity
        if emotion_score > 0
        else intensity * 0.25
    )

    content_score = 0.0

    if emotion in item["emotions"]:
        content_score += 0.60

    matching_tags = sum(
        1
        for tag in item["tags"]
        if tag in preferences
    )

    if preferences:
        content_score += min(
            0.40,
            matching_tags * 0.20
        )

    content_score = min(
        1.0,
        content_score
    )

    # -----------------------------------------------------
    # FINAL RECOMMENDATION SCORE
    # -----------------------------------------------------

    final_score = (
        0.25 * emotion_score
        + 0.15 * intensity_score
        + 0.15 * preference_score
        + 0.15 * content_score
        + 0.15 * interaction_score
        + 0.15 * trend_score
    )

    # Avoid recommending already consumed content
    final_score *= (
        0.70 + 0.30 * history_score
    )

    return {
        "emotion_score": emotion_score,
        "intensity_score": intensity_score,
        "preference_score": preference_score,
        "content_score": content_score,
        "interaction_score": interaction_score,
        "history_score": history_score,
        "trend_score": trend_score,
        "final_score": final_score
    }


# ---------------------------------------------------------
# GENERATE RECOMMENDATIONS
# ---------------------------------------------------------

def generate_recommendations(
    emotion,
    intensity,
    preferences,
    interactions,
    history,
    trend
):
    results = []

    for item in WELLNESS_CONTENT:

        features = calculate_features(
            item,
            emotion,
            intensity,
            preferences,
            interactions,
            history,
            trend
        )

        results.append({
            "id": item["id"],
            "title": item["title"],
            "features": features
        })

    results.sort(
        key=lambda x: x["features"]["final_score"],
        reverse=True
    )

    return results


# ---------------------------------------------------------
# EXPLANATION GENERATION
# ---------------------------------------------------------

def generate_explanation(
    item,
    features,
    emotion,
    intensity,
    preferences,
    interactions,
    history,
    trend
):
    reasons = []

    # Emotion
    if features["emotion_score"] > 0:
        reasons.append(
            f"matches your detected emotion ({emotion})"
        )

    # Intensity
    if features["intensity_score"] >= 0.70:
        reasons.append(
            f"your emotional intensity is high ({intensity:.2f})"
        )
    elif features["intensity_score"] >= 0.40:
        reasons.append(
            f"your emotional intensity is moderate ({intensity:.2f})"
        )

    # Preference
    matched_preferences = [
        preference
        for preference in preferences
        if (
            preference in item["tags"]
            or preference in item["title"].lower()
        )
    ]

    if matched_preferences:
        reasons.append(
            "matches your preference for "
            + ", ".join(matched_preferences)
        )

    # Previous interaction
    if features["interaction_score"] > 0:
        reasons.append(
            "you interacted with this content before"
        )

    # History
    if features["history_score"] > 0:
        reasons.append(
            "this content has not appeared in your recommendation history"
        )
    else:
        reasons.append(
            "recommendation history was considered"
        )

    # Trend
    if features["trend_score"] > 0:
        reasons.append(
            f"matches your recent emotional trend ({trend})"
        )

    # Semantic/content relevance
    if features["content_score"] >= 0.60:
        reasons.append(
            "has strong content relevance to your emotional state"
        )

    if not reasons:
        reasons.append(
            "received a recommendation score based on the current user state"
        )

    explanation = (
        f"{item['title']} was recommended because "
        + "; ".join(reasons)
        + "."
    )

    return explanation


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_explanations(results):

    checks = {}

    checks["Results Present"] = (
        len(results) > 0
    )

    explanations = [
        result.get("explanation", "")
        for result in results
    ]

    checks["Explanation Present"] = all(
        len(explanation.strip()) > 0
        for explanation in explanations
    )

    checks["Dynamic Reasons"] = len(
        set(explanations)
    ) > 1

    ids = [
        result["id"]
        for result in results
    ]

    checks["No Duplicates"] = (
        len(ids) == len(set(ids))
    )

    scores = [
        result["features"]["final_score"]
        for result in results
    ]

    checks["Ranking Order"] = all(
        scores[i] >= scores[i + 1]
        for i in range(len(scores) - 1)
    )

    return checks


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Milestone 3 - "
            "Task 8 Recommendation Explainability"
        )
    )

    parser.add_argument(
        "--emotion",
        default="Fear",
        help="Current dominant emotion"
    )

    parser.add_argument(
        "--intensity",
        type=float,
        default=0.7736,
        help="Current emotional intensity"
    )

    parser.add_argument(
        "--preferences",
        default="breathing,mindfulness",
        help="Comma-separated user preferences"
    )

    parser.add_argument(
        "--interactions",
        default="W06,W08",
        help="Previous interactions"
    )

    parser.add_argument(
        "--history",
        default="W03,W07",
        help="Recommendation history"
    )

    parser.add_argument(
        "--trend",
        default="Fear",
        help="Recent emotional trend"
    )

    args = parser.parse_args()

    emotion = args.emotion.title()

    preferences = parse_list(
        args.preferences
    )

    interactions = [
        item.strip().upper()
        for item in args.interactions.split(",")
        if item.strip()
    ]

    history = [
        item.strip().upper()
        for item in args.history.split(",")
        if item.strip()
    ]

    trend = args.trend.title()

    print("\n" + "=" * 60)
    print(
        "TASK 8 - RECOMMENDATION "
        "EXPLAINABILITY"
    )
    print("=" * 60)

    print("\nUSER STATE")
    print("-" * 45)

    print(
        f"Detected emotion : {emotion}"
    )

    print(
        f"Intensity        : {args.intensity:.4f}"
    )

    print(
        f"Preferences      : "
        f"{', '.join(preferences)}"
    )

    print(
        f"Previous interactions : "
        f"{', '.join(interactions)}"
    )

    print(
        f"Recommendation history : "
        f"{', '.join(history)}"
    )

    print(
        f"Emotional trend : {trend}"
    )

    # -----------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # -----------------------------------------------------

    results = generate_recommendations(
        emotion,
        args.intensity,
        preferences,
        interactions,
        history,
        trend
    )

    # -----------------------------------------------------
    # GENERATE EXPLANATIONS
    # -----------------------------------------------------

    for result in results:

        item = next(
            content
            for content in WELLNESS_CONTENT
            if content["id"] == result["id"]
        )

        result["explanation"] = generate_explanation(
            item,
            result["features"],
            emotion,
            args.intensity,
            preferences,
            interactions,
            history,
            trend
        )

    # -----------------------------------------------------
    # DISPLAY TOP RECOMMENDATIONS
    # -----------------------------------------------------

    print(
        "\nEXPLAINABLE RECOMMENDATIONS"
    )

    print("-" * 60)

    for i, result in enumerate(
        results[:5],
        start=1
    ):

        features = result["features"]

        print(
            f"\n{i}. {result['title']} "
            f"[{result['id']}]"
        )

        print(
            f"   Final score: "
            f"{features['final_score']:.4f}"
        )

        print(
            f"   Emotion relevance: "
            f"{features['emotion_score']:.4f}"
        )

        print(
            f"   Intensity relevance: "
            f"{features['intensity_score']:.4f}"
        )

        print(
            f"   Preference match: "
            f"{features['preference_score']:.4f}"
        )

        print(
            f"   Content relevance: "
            f"{features['content_score']:.4f}"
        )

        print(
            f"   Previous interaction: "
            f"{features['interaction_score']:.4f}"
        )

        print(
            f"   History factor: "
            f"{features['history_score']:.4f}"
        )

        print(
            f"   Trend relevance: "
            f"{features['trend_score']:.4f}"
        )

        print(
            "   WHY: "
            + result["explanation"]
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    checks = validate_explanations(
        results
    )

    print("\nVALIDATION")

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # -----------------------------------------------------
    # DYNAMIC EXPLANATION TEST
    # -----------------------------------------------------

    changed_preferences = [
        "social",
        "walking"
    ]

    changed_interactions = [
        "W11",
        "W06"
    ]

    changed_history = [
        "W01",
        "W02"
    ]

    changed_trend = "Sadness"

    changed_results = generate_recommendations(
        emotion,
        args.intensity,
        changed_preferences,
        changed_interactions,
        changed_history,
        changed_trend
    )

    for result in changed_results:

        item = next(
            content
            for content in WELLNESS_CONTENT
            if content["id"] == result["id"]
        )

        result["explanation"] = generate_explanation(
            item,
            result["features"],
            emotion,
            args.intensity,
            changed_preferences,
            changed_interactions,
            changed_history,
            changed_trend
        )

    original_top = results[0]["id"]
    changed_top = changed_results[0]["id"]

    explanations_changed = (
        results[0]["explanation"]
        != changed_results[0]["explanation"]
    )

    print(
        "\nDYNAMIC EXPLANATION VERIFICATION"
    )

    print("-" * 60)

    print(
        f"Original top recommendation : "
        f"{original_top}"
    )

    print(
        f"Changed profile top recommendation : "
        f"{changed_top}"
    )

    print(
        "Explanation changes with user context: "
        + (
            "PASS"
            if explanations_changed
            else "FAIL"
        )
    )

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    all_checks_passed = (
        all(checks.values())
        and explanations_changed
    )

    print("\n" + "=" * 60)

    if all_checks_passed:
        print(
            "TASK 8 OVERALL: "
            "ALL CHECKS PASSED"
        )
    else:
        print(
            "TASK 8 OVERALL: "
            "SOME CHECKS FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()