import argparse
import time
from statistics import mean


# =========================================================
# WELLNESS CONTENT
# =========================================================

WELLNESS_CONTENT = [
    {
        "id": "W01",
        "title": "Deep Breathing",
        "emotions": ["Fear"],
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


# =========================================================
# CONTROLLED VALIDATION DATASET
# =========================================================

TEST_CASES = [
    {
        "name": "Fear / Exam Stress",
        "emotion": "Fear",
        "preferences": ["breathing", "mindfulness"],
        "relevant": {"W01", "W02", "W05", "W06", "W08", "W09", "W11"}
    },
    {
        "name": "Sadness / Emotional Low",
        "emotion": "Sadness",
        "preferences": ["support", "reflection"],
        "relevant": {"W02", "W05", "W06", "W08", "W09", "W10", "W11"}
    },
    {
        "name": "Joy / Positive Mood",
        "emotion": "Joy",
        "preferences": ["motivation", "music"],
        "relevant": {"W03", "W07", "W12"}
    },
    {
        "name": "Anger / Frustration",
        "emotion": "Anger",
        "preferences": ["journaling", "regulation"],
        "relevant": {"W04", "W05", "W06", "W09", "W10"}
    }
]


# =========================================================
# BASELINE RECOMMENDER
# =========================================================

def baseline_recommendation(emotion):
    results = []

    for item in WELLNESS_CONTENT:

        score = 0.0

        if emotion in item["emotions"]:
            score = 1.0

        results.append(
            (item["id"], score)
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item_id
        for item_id, score in results[:5]
    ]


# =========================================================
# ADVANCED RECOMMENDER
# =========================================================

def advanced_recommendation(
    emotion,
    preferences
):
    results = []

    for item in WELLNESS_CONTENT:

        emotion_score = (
            1.0
            if emotion in item["emotions"]
            else 0.0
        )

        preference_matches = sum(
            1
            for preference in preferences
            if preference in item["tags"]
        )

        preference_score = min(
            1.0,
            preference_matches * 0.5
        )

        content_score = 0.0

        if emotion in item["emotions"]:
            content_score += 0.60

        content_score += (
            preference_score * 0.40
        )

        final_score = (
            0.50 * emotion_score
            + 0.25 * preference_score
            + 0.25 * content_score
        )

        results.append(
            {
                "id": item["id"],
                "score": final_score
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        item["id"]
        for item in results[:5]
    ]


# =========================================================
# METRICS
# =========================================================

def precision_at_k(
    recommended,
    relevant,
    k=5
):
    recommended_k = recommended[:k]

    if not recommended_k:
        return 0.0

    hits = sum(
        1
        for item in recommended_k
        if item in relevant
    )

    return hits / len(recommended_k)


def recall_at_k(
    recommended,
    relevant,
    k=5
):
    recommended_k = recommended[:k]

    if not relevant:
        return 0.0

    hits = sum(
        1
        for item in recommended_k
        if item in relevant
    )

    return hits / len(relevant)


def f1_at_k(
    precision,
    recall
):
    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# =========================================================
# RANKING QUALITY
# =========================================================

def ranking_quality(
    recommended,
    relevant
):
    score = 0.0

    for position, item in enumerate(
        recommended,
        start=1
    ):
        if item in relevant:
            score += 1 / position

    return score


# =========================================================
# ACCEPTANCE RATE
# =========================================================

def acceptance_rate(
    recommended,
    relevant
):
    if not recommended:
        return 0.0

    accepted = sum(
        1
        for item in recommended
        if item in relevant
    )

    return accepted / len(recommended)


# =========================================================
# DIVERSITY
# =========================================================

def recommendation_diversity(
    recommended
):
    selected = [
        item
        for item in WELLNESS_CONTENT
        if item["id"] in recommended
    ]

    if not selected:
        return 0.0

    unique_emotions = set()
    unique_tags = set()

    for item in selected:
        unique_emotions.update(
            item["emotions"]
        )

        unique_tags.update(
            item["tags"]
        )

    emotion_diversity = (
        len(unique_emotions)
        / len(recommended)
    )

    tag_diversity = (
        len(unique_tags)
        / len(recommended)
    )

    return min(
        1.0,
        (emotion_diversity + tag_diversity) / 2
    )


# =========================================================
# RUN EVALUATION
# =========================================================

def evaluate_recommender(
    recommender,
    name
):
    precisions = []
    recalls = []
    f1_scores = []
    ranking_scores = []
    acceptance_scores = []
    diversity_scores = []
    response_times = []

    print(
        f"\n{name} EVALUATION"
    )
    print("-" * 60)

    for case in TEST_CASES:

        start = time.perf_counter()

        recommendations = recommender(
            case["emotion"],
            case["preferences"]
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        precision = precision_at_k(
            recommendations,
            case["relevant"]
        )

        recall = recall_at_k(
            recommendations,
            case["relevant"]
        )

        f1 = f1_at_k(
            precision,
            recall
        )

        ranking = ranking_quality(
            recommendations,
            case["relevant"]
        )

        acceptance = acceptance_rate(
            recommendations,
            case["relevant"]
        )

        diversity = recommendation_diversity(
            recommendations
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ranking_scores.append(ranking)
        acceptance_scores.append(acceptance)
        diversity_scores.append(diversity)
        response_times.append(elapsed)

        print(
            f"\n{case['name']}"
        )

        print(
            "Recommendations: "
            + " > ".join(recommendations)
        )

        print(
            f"Precision@5 : {precision:.4f}"
        )

        print(
            f"Recall@5    : {recall:.4f}"
        )

        print(
            f"F1@5        : {f1:.4f}"
        )

        print(
            f"Ranking      : {ranking:.4f}"
        )

        print(
            f"Acceptance   : {acceptance:.4f}"
        )

        print(
            f"Diversity    : {diversity:.4f}"
        )

        print(
            f"Response ms  : {elapsed:.4f}"
        )

    return {
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1_scores),
        "ranking": mean(ranking_scores),
        "acceptance": mean(acceptance_scores),
        "diversity": mean(diversity_scores),
        "response_time": mean(response_times)
    }


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Milestone 3 - "
            "Task 9 Advanced ML Validation"
        )
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(
        "TASK 9 - ADVANCED ML VALIDATION "
        "& PERFORMANCE TESTING"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------

    baseline = evaluate_recommender(
        lambda emotion, preferences:
            baseline_recommendation(emotion),
        "BASELINE"
    )

    # -----------------------------------------------------
    # ADVANCED
    # -----------------------------------------------------

    advanced = evaluate_recommender(
        advanced_recommendation,
        "ADVANCED RECOMMENDER"
    )

    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BASELINE VS ADVANCED"
    )

    print(
        "=" * 60
    )

    metrics = [
        ("Precision@5", "precision"),
        ("Recall@5", "recall"),
        ("F1@5", "f1"),
        ("Ranking Quality", "ranking"),
        ("Acceptance Rate", "acceptance"),
        ("Diversity", "diversity"),
        ("Response Time (ms)", "response_time")
    ]

    for label, key in metrics:

        print(
            f"{label:<25} "
            f"Baseline={baseline[key]:.4f} "
            f"Advanced={advanced[key]:.4f}"
        )

    # -----------------------------------------------------
    # VALIDATION CHECKS
    # -----------------------------------------------------

    print(
        "\nVALIDATION"
    )

    precision_pass = (
        advanced["precision"] > 0
    )

    recall_pass = (
        advanced["recall"] > 0
    )

    f1_pass = (
        advanced["f1"] > 0
    )

    ranking_pass = (
        advanced["ranking"] > 0
    )

    acceptance_pass = (
        advanced["acceptance"] > 0
    )

    diversity_pass = (
        advanced["diversity"] > 0
    )

    response_pass = (
        advanced["response_time"] < 1000
    )

    checks = {
        "Precision@5": precision_pass,
        "Recall@5": recall_pass,
        "F1@5": f1_pass,
        "Ranking Quality": ranking_pass,
        "Acceptance Rate": acceptance_pass,
        "Diversity": diversity_pass,
        "Response Time": response_pass
    }

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # -----------------------------------------------------
    # ADVANCED MODEL IMPROVEMENT
    # -----------------------------------------------------

    advanced_improved = (
        advanced["precision"]
        >= baseline["precision"]
        and
        advanced["recall"]
        >= baseline["recall"]
    )

    print(
        "\nAdvanced recommender maintains or "
        "improves Precision/Recall: "
        + (
            "PASS"
            if advanced_improved
            else "FAIL"
        )
    )

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    all_passed = (
        all(checks.values())
        and advanced_improved
    )

    print(
        "\n" + "=" * 60
    )

    if all_passed:
        print(
            "TASK 9 OVERALL: "
            "ALL CHECKS PASSED"
        )
    else:
        print(
            "TASK 9 OVERALL: "
            "SOME CHECKS FAILED"
        )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()