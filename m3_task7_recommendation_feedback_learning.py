import argparse
from collections import defaultdict


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
        "title": "Grounding 5-4-3-2-1",
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
# FEEDBACK STORAGE
# ---------------------------------------------------------

feedback_history = []


def store_feedback(
    recommendation_id,
    feedback_type,
    rating=None
):
    feedback = {
        "recommendation_id": recommendation_id,
        "feedback": feedback_type,
        "rating": rating
    }

    feedback_history.append(feedback)


# ---------------------------------------------------------
# FEEDBACK LEARNING
# ---------------------------------------------------------

def calculate_feedback_score(recommendation_id):
    score = 0.0

    for feedback in feedback_history:

        if feedback["recommendation_id"] != recommendation_id:
            continue

        feedback_type = feedback["feedback"]

        if feedback_type == "accepted":
            score += 0.30

        elif feedback_type == "rejected":
            score -= 0.30

        elif feedback_type == "viewed":
            score += 0.05

        elif feedback_type == "rating":
            rating = feedback["rating"]

            if rating is not None:
                score += (
                    (rating - 3) / 10
                )

    return score


# ---------------------------------------------------------
# BASE RECOMMENDATION SCORE
# ---------------------------------------------------------

def calculate_base_score(item, emotion):

    score = 0.0

    if emotion in item["emotions"]:
        score += 0.60

    if "calm" in item["tags"]:
        score += 0.15

    if "stress" in item["tags"]:
        score += 0.15

    if "mindfulness" in item["tags"]:
        score += 0.10

    return min(score, 1.0)


# ---------------------------------------------------------
# GENERATE RECOMMENDATIONS
# ---------------------------------------------------------

def generate_recommendations(emotion):

    results = []

    for item in WELLNESS_CONTENT:

        base_score = calculate_base_score(
            item,
            emotion
        )

        feedback_score = calculate_feedback_score(
            item["id"]
        )

        final_score = base_score + feedback_score

        final_score = max(
            0.0,
            min(1.0, final_score)
        )

        results.append({
            "id": item["id"],
            "title": item["title"],
            "base_score": base_score,
            "feedback_score": feedback_score,
            "final_score": final_score
        })

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results


# ---------------------------------------------------------
# PRINT RECOMMENDATIONS
# ---------------------------------------------------------

def print_recommendations(
    results,
    title
):

    print("\n" + title)
    print("-" * 55)

    for i, item in enumerate(
        results[:5],
        start=1
    ):

        print(
            f"{i}. {item['title']} "
            f"[{item['id']}] "
            f"base={item['base_score']:.4f} "
            f"feedback={item['feedback_score']:+.4f} "
            f"final={item['final_score']:.4f}"
        )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_results(results):

    checks = {}

    checks["Results Present"] = (
        len(results) > 0
    )

    ids = [
        item["id"]
        for item in results
    ]

    checks["No Duplicates"] = (
        len(ids) == len(set(ids))
    )

    scores = [
        item["final_score"]
        for item in results
    ]

    checks["Ranking Order"] = all(
        scores[i] >= scores[i + 1]
        for i in range(len(scores) - 1)
    )

    return checks


# ---------------------------------------------------------
# RANKING COMPARISON
# ---------------------------------------------------------

def get_ranking(results):

    return [
        item["id"]
        for item in results[:5]
    ]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Milestone 3 - "
            "Task 7 Recommendation Feedback Learning"
        )
    )

    parser.add_argument(
        "--emotion",
        default="Fear",
        help="Current dominant emotion"
    )

    args = parser.parse_args()

    emotion = args.emotion.title()

    print("\n" + "=" * 60)
    print(
        "TASK 7 - RECOMMENDATION "
        "FEEDBACK LEARNING"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # BEFORE FEEDBACK
    # -----------------------------------------------------

    before_results = generate_recommendations(
        emotion
    )

    print_recommendations(
        before_results,
        "RECOMMENDATIONS BEFORE FEEDBACK"
    )

    original_ranking = get_ranking(
        before_results
    )

    # -----------------------------------------------------
    # USER FEEDBACK
    # -----------------------------------------------------

    print("\nUSER FEEDBACK")
    print("-" * 55)

    print(
        "User accepted W06 - Mindful Walk"
    )

    store_feedback(
        "W06",
        "accepted"
    )

    print(
        "User rejected W01 - Deep Breathing"
    )

    store_feedback(
        "W01",
        "rejected"
    )

    print(
        "User rated W02 - Grounding 5-4-3-2-1 as 5/5"
    )

    store_feedback(
        "W02",
        "rating",
        5
    )

    print(
        "Feedback records stored:",
        len(feedback_history)
    )

    # -----------------------------------------------------
    # AFTER FEEDBACK
    # -----------------------------------------------------

    after_results = generate_recommendations(
        emotion
    )

    print_recommendations(
        after_results,
        "RECOMMENDATIONS AFTER FEEDBACK"
    )

    updated_ranking = get_ranking(
        after_results
    )

    # -----------------------------------------------------
    # FEEDBACK EFFECT
    # -----------------------------------------------------

    print("\nFEEDBACK LEARNING EFFECT")
    print("-" * 55)

    for recommendation_id in [
        "W06",
        "W01",
        "W02"
    ]:

        feedback_score = calculate_feedback_score(
            recommendation_id
        )

        print(
            f"{recommendation_id} "
            f"feedback adjustment: "
            f"{feedback_score:+.4f}"
        )

    ranking_changed = (
        original_ranking != updated_ranking
    )

    print(
        "\nOriginal ranking : "
        + " > ".join(original_ranking)
    )

    print(
        "Updated ranking  : "
        + " > ".join(updated_ranking)
    )

    print(
        "Ranking changed after feedback: "
        + (
            "PASS"
            if ranking_changed
            else "FAIL"
        )
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    checks = validate_results(
        after_results
    )

    print("\nVALIDATION")

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    feedback_stored = (
        len(feedback_history) >= 3
    )

    print(
        f"Feedback Storage: "
        f"{'PASS' if feedback_stored else 'FAIL'}"
    )

    feedback_applied = any(
        item["feedback_score"] != 0
        for item in after_results
    )

    print(
        f"Feedback Applied to Ranking: "
        f"{'PASS' if feedback_applied else 'FAIL'}"
    )

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    all_checks_passed = (
        all(checks.values())
        and feedback_stored
        and feedback_applied
        and ranking_changed
    )

    print("\n" + "=" * 60)

    if all_checks_passed:
        print(
            "TASK 7 OVERALL: "
            "ALL CHECKS PASSED"
        )
    else:
        print(
            "TASK 7 OVERALL: "
            "SOME CHECKS FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()