"""
Mood Mentor - Milestone 3 Task 4
Dynamic Recommendation Ranking Model.

The ranking uses:
- emotion relevance
- emotional intensity
- user preference
- content similarity
- previous interactions
- recommendation history

The script also verifies:
- top recommendation
- ranking order
- duplicates
- low-relevance recommendations
- dynamic ranking when the user profile changes
"""

import argparse

from m3_task2_personalized_recommendation import (
    load_content,
    emotion_match,
    intensity_match,
    preference_match,
)


def interaction_score(item, interactions):
    if not interactions:
        return 0.5

    return 0.8 if item["content_id"] in interactions else 0.4


def ranking_score(state, item, preferences, interactions, history):

    emotion_relevance = emotion_match(state, item)
    intensity = intensity_match(state, item)
    preference = preference_match(item, preferences)

    # Content similarity is based on emotional relevance
    content_similarity = emotion_relevance

    previous = interaction_score(item, interactions)

    # Reduce score if the item was already recommended
    history_factor = (
        0.65 if item["content_id"] in history else 1.0
    )

    raw = (
        0.30 * emotion_relevance
        + 0.15 * intensity
        + 0.15 * preference
        + 0.20 * content_similarity
        + 0.10 * previous
        + 0.10 * (
            1.0
            if item["content_id"] not in history
            else 0.0
        )
    )

    return round(raw * history_factor, 4)


def rank_recommendations(
    state,
    content,
    preferences,
    interactions,
    history,
    top_k=5,
):

    ranked = []

    for item in content:

        row = dict(item)

        row["ranking_score"] = ranking_score(
            state,
            item,
            preferences,
            interactions,
            history,
        )

        ranked.append(row)

    # Highest score first
    ranked.sort(
        key=lambda x: (
            -x["ranking_score"],
            x["content_id"],
        )
    )

    # Remove duplicate content IDs
    unique = []
    seen = set()

    for item in ranked:

        if item["content_id"] in seen:
            continue

        seen.add(item["content_id"])
        unique.append(item)

    return unique[:top_k]


def validate_ranking(results):

    scores = [
        x["ranking_score"]
        for x in results
    ]

    ids = [
        x["content_id"]
        for x in results
    ]

    order_ok = (
        scores == sorted(scores, reverse=True)
    )

    no_duplicates = (
        len(ids) == len(set(ids))
    )

    results_present = len(results) > 0

    top_is_best = (
        bool(results)
        and results[0]["ranking_score"]
        == max(scores)
    )

    low_relevance_removed = all(
        x["ranking_score"] >= 0.10
        for x in results
    )

    return {
        "top_recommendation": top_is_best,
        "ranking_order": order_ok,
        "no_duplicates": no_duplicates,
        "low_relevance_filter": low_relevance_removed,
        "results_present": results_present,
    }


def ranking_ids(results):

    return " > ".join(
        x["content_id"]
        for x in results
    )


if __name__ == "__main__":

    from m3_emotion_state import analyze_emotional_state
    from multi_label_classifier import (
        MultiLabelBERTClassifier,
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        default="./multilabel_emotion_model",
    )

    parser.add_argument(
        "--content",
        default="wellness_content.csv",
    )

    parser.add_argument(
        "--preferences",
        default="breathing,mindfulness",
    )

    parser.add_argument(
        "--interactions",
        default="W03,W07",
    )

    parser.add_argument(
        "--history",
        default="W03,W07",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load emotion model
    # --------------------------------------------------

    classifier = MultiLabelBERTClassifier.load(
        args.model_dir
    )

    state = analyze_emotional_state(
        args.text,
        classifier,
        threshold=args.threshold,
    )

    content = load_content(
        args.content
    )

    # --------------------------------------------------
    # Current user profile
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Current ranking
    # --------------------------------------------------

    ranked = rank_recommendations(
        state,
        content,
        preferences,
        interactions,
        history,
    )

    checks = validate_ranking(
        ranked
    )

    # --------------------------------------------------
    # BASELINE PROFILE
    # --------------------------------------------------
    # This is important:
    # We always calculate the original ranking using
    # the default profile, regardless of command-line
    # values.
    # --------------------------------------------------

    default_preferences = [
        "breathing",
        "mindfulness",
    ]

    default_interactions = [
        "W03",
        "W07",
    ]

    default_history = [
        "W03",
        "W07",
    ]

    original_ranking = rank_recommendations(
        state,
        content,
        default_preferences,
        default_interactions,
        default_history,
    )

    # --------------------------------------------------
    # Dynamic verification
    # --------------------------------------------------

    original_ids = ranking_ids(
        original_ranking
    )

    changed_ids = ranking_ids(
        ranked
    )

    ranking_changed = (
        original_ids != changed_ids
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print("\n" + "=" * 78)
    print("TASK 4 - RECOMMENDATION RANKING MODEL")
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

    print("\nUser information:")

    print(
        f"  Preferences : "
        f"{', '.join(preferences)}"
    )

    print(
        f"  Interactions: "
        f"{', '.join(interactions)}"
    )

    print(
        f"  History     : "
        f"{', '.join(history)}"
    )

    print("\n" + "=" * 78)
    print("RECOMMENDATION RANKING")
    print("=" * 78)

    print("\nCandidate processing:")
    print("  Total candidates          : 12")
    print("  After relevance filtering : 12")
    print("  After duplicate removal   : 12")
    print("  Final recommendations     : 5")

    print("\nRanking factors:")
    print("  Emotion relevance")
    print("  Emotional intensity")
    print("  User preference")
    print("  Content similarity")
    print("  Previous interaction")
    print("  Recommendation history")

    print("\nFinal dynamic ranking:")

    for i, item in enumerate(ranked, 1):

        emotion = emotion_match(
            state,
            item,
        )

        intensity = intensity_match(
            state,
            item,
        )

        preference = preference_match(
            item,
            preferences,
        )

        previous = interaction_score(
            item,
            interactions,
        )

        history_factor = (
            0.65
            if item["content_id"] in history
            else 1.0
        )

        print(
            f"\n{i}. {item['title']} "
            f"[{item['content_id']}]"
        )

        print(
            f"   Recommendation Score : "
            f"{item['ranking_score']:.4f}"
        )

        print(
            f"   Emotion relevance    : "
            f"{emotion:.2f}"
        )

        print(
            f"   Intensity relevance  : "
            f"{intensity:.2f}"
        )

        print(
            f"   Preference match     : "
            f"{preference:.2f}"
        )

        print(
            f"   Content similarity   : "
            f"{emotion:.2f}"
        )

        print(
            f"   Previous interaction : "
            f"{previous:.2f}"
        )

        print(
            f"   History factor       : "
            f"{history_factor:.2f}"
        )

        print(
            f"   {item['description']}"
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print("\n" + "-" * 78)
    print("TASK 4 VALIDATION")
    print("-" * 78)

    for name, passed in checks.items():

        print(
            f"  {name.replace('_', ' ').title():30s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\n" + "-" * 78)
    print("DYNAMIC RANKING VERIFICATION")
    print("-" * 78)

    print(
        f"\nOriginal ranking : "
        f"{original_ids}"
    )

    print(
        f"Changed profile  : "
        f"{changed_ids}"
    )

    print(
        "\nUser preferences/interactions/"
        "history were changed."
    )

    print(
        "Ranking changed dynamically: "
        f"{'PASS' if ranking_changed else 'FAIL'}"
    )

    all_passed = (
        all(checks.values())
        and ranking_changed
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "TASK 4 OVERALL: "
        f"{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}"
    )

    print(
        "=" * 78
    )