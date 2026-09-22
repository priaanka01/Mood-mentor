import argparse
from collections import Counter
from statistics import mean

from m3_emotion_state import analyze_emotional_state
from multi_label_classifier import MultiLabelBERTClassifier


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
# PARSE HISTORICAL EMOTION DATA
# Format:
# Fear:0.70,Sadness:0.65,Fear:0.80
# ---------------------------------------------------------

def parse_history(history_text):
    records = []

    for item in history_text.split(","):
        item = item.strip()

        if ":" not in item:
            continue

        emotion, intensity = item.split(":", 1)

        try:
            intensity = float(intensity)
        except ValueError:
            continue

        records.append({
            "emotion": emotion.strip().title(),
            "intensity": max(0.0, min(1.0, intensity))
        })

    return records


# ---------------------------------------------------------
# TREND ANALYSIS
# ---------------------------------------------------------

def analyze_history(records):
    if not records:
        return None

    emotions = [r["emotion"] for r in records]
    intensities = [r["intensity"] for r in records]

    frequency = Counter(emotions)

    dominant_emotion = frequency.most_common(1)[0][0]

    split_point = max(1, len(intensities) // 2)

    first_half = intensities[:split_point]
    second_half = intensities[split_point:]

    first_avg = mean(first_half)
    second_avg = mean(second_half)

    intensity_change = second_avg - first_avg

    if intensity_change > 0.05:
        intensity_trend = "Increasing"
    elif intensity_change < -0.05:
        intensity_trend = "Decreasing"
    else:
        intensity_trend = "Stable"

    negative_emotions = {
        "Fear",
        "Sadness",
        "Anger",
        "Disgust"
    }

    positive_emotions = {
        "Joy",
        "Surprise"
    }

    negative_count = sum(
        1 for e in emotions
        if e in negative_emotions
    )

    positive_count = sum(
        1 for e in emotions
        if e in positive_emotions
    )

    if negative_count > positive_count:
        polarity_trend = "Negative"
    elif positive_count > negative_count:
        polarity_trend = "Positive"
    else:
        polarity_trend = "Balanced"

    repeated_emotions = [
        emotion
        for emotion, count in frequency.items()
        if count >= 2
    ]

    recent_state = records[-1]["emotion"]

    return {
        "frequency": frequency,
        "dominant_emotion": dominant_emotion,
        "first_avg": first_avg,
        "second_avg": second_avg,
        "intensity_change": intensity_change,
        "intensity_trend": intensity_trend,
        "polarity_trend": polarity_trend,
        "repeated_emotions": repeated_emotions,
        "recent_state": recent_state
    }


# ---------------------------------------------------------
# HISTORY INFLUENCE
# ---------------------------------------------------------

def calculate_history_influence(item, trend):
    score = 0.0

    # 1. Dominant historical emotion
    if trend["dominant_emotion"] in item["emotions"]:
        score += 0.40

    # 2. Recent emotional state
    if trend["recent_state"] in item["emotions"]:
        score += 0.25

    # 3. Repeated emotional patterns
    repeated_matches = sum(
        1
        for emotion in trend["repeated_emotions"]
        if emotion in item["emotions"]
    )

    score += min(
        0.20,
        repeated_matches * 0.10
    )

    # 4. Negative emotional trend
    if trend["polarity_trend"] == "Negative":

        stress_tags = {
            "calm",
            "stress",
            "relaxation",
            "grounding",
            "mindfulness",
            "support"
        }

        if any(
            tag in stress_tags
            for tag in item["tags"]
        ):
            score += 0.10

    # 5. Increasing intensity
    if trend["intensity_trend"] == "Increasing":

        regulation_tags = {
            "calm",
            "stress",
            "relaxation",
            "grounding",
            "regulation",
            "mindfulness"
        }

        if any(
            tag in regulation_tags
            for tag in item["tags"]
        ):
            score += 0.10

    return min(score, 1.0)


# ---------------------------------------------------------
# RECOMMENDATIONS BASED ON EMOTIONAL HISTORY
# ---------------------------------------------------------

def generate_recommendations(trend):
    results = []

    for item in WELLNESS_CONTENT:

        history_score = calculate_history_influence(
            item,
            trend
        )

        results.append({
            "id": item["id"],
            "title": item["title"],
            "score": history_score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ---------------------------------------------------------
# PRINT TREND ANALYSIS
# ---------------------------------------------------------

def print_trend_analysis(trend):

    print("\nEMOTIONAL TREND ANALYSIS")
    print("-" * 45)

    print("Emotion frequency:")

    for emotion, count in trend["frequency"].most_common():
        print(
            f"  {emotion}: {count}"
        )

    print(
        f"\nDominant historical emotion : "
        f"{trend['dominant_emotion']}"
    )

    print(
        f"Average intensity - early  : "
        f"{trend['first_avg']:.4f}"
    )

    print(
        f"Average intensity - recent : "
        f"{trend['second_avg']:.4f}"
    )

    print(
        f"Intensity change            : "
        f"{trend['intensity_change']:+.4f}"
    )

    print(
        f"Intensity trend             : "
        f"{trend['intensity_trend']}"
    )

    print(
        f"Positive/Negative trend     : "
        f"{trend['polarity_trend']}"
    )

    repeated = (
        ", ".join(trend["repeated_emotions"])
        if trend["repeated_emotions"]
        else "None"
    )

    print(
        f"Repeated emotional patterns : "
        f"{repeated}"
    )

    print(
        f"Recent emotional state      : "
        f"{trend['recent_state']}"
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_results(results):

    checks = {}

    checks["Results Present"] = len(results) > 0

    scores = [
        r["score"]
        for r in results
    ]

    checks["Ranking Order"] = all(
        scores[i] >= scores[i + 1]
        for i in range(len(scores) - 1)
    )

    ids = [
        r["id"]
        for r in results
    ]

    checks["No Duplicates"] = (
        len(ids) == len(set(ids))
    )

    checks["History Influence Present"] = any(
        r["score"] > 0
        for r in results
    )

    return checks


# ---------------------------------------------------------
# RANKING IDS
# ---------------------------------------------------------

def ranking_ids(results):
    return [
        r["id"]
        for r in results[:5]
    ]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Milestone 3 - "
            "Task 6 Emotional Trend Tracking"
        )
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Current user text"
    )

    parser.add_argument(
        "--history",
        required=True,
        help=(
            "Historical emotions in "
            "Emotion:Intensity format"
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # LOAD EXISTING BERT MODEL
    # -----------------------------------------------------

    classifier = MultiLabelBERTClassifier.load(
        "./multilabel_emotion_model"
    )

    # -----------------------------------------------------
    # CURRENT EMOTIONAL STATE
    # -----------------------------------------------------

    current_state = analyze_emotional_state(
        args.text,
        classifier
    )

    print("\n" + "=" * 60)
    print(
        "TASK 6 - EMOTIONAL TREND "
        "& USER STATE TRACKING"
    )
    print("=" * 60)

    print("\nCURRENT EMOTIONAL STATE")
    print("-" * 45)

    print(
        f"Current dominant emotion : "
        f"{current_state.dominant_emotion}"
    )

    print(
        f"Current intensity         : "
        f"{current_state.emotional_intensity:.4f}"
    )

    print(
        f"Current polarity          : "
        f"{current_state.polarity}"
    )

    # -----------------------------------------------------
    # HISTORICAL DATA
    # -----------------------------------------------------

    records = parse_history(
        args.history
    )

    print("\nHistorical records:")

    for i, record in enumerate(
        records,
        start=1
    ):
        print(
            f"  {i}. {record['emotion']} "
            f"(intensity={record['intensity']:.2f})"
        )

    trend = analyze_history(records)

    if trend is None:
        print(
            "\nNo valid historical data found."
        )
        return

    # -----------------------------------------------------
    # TREND ANALYSIS
    # -----------------------------------------------------

    print_trend_analysis(trend)

    # -----------------------------------------------------
    # TREND-INFLUENCED RECOMMENDATIONS
    # -----------------------------------------------------

    results = generate_recommendations(
        trend
    )

    print(
        "\nTREND-INFLUENCED "
        "RECOMMENDATIONS"
    )

    print("-" * 45)

    for i, result in enumerate(
        results[:5],
        start=1
    ):
        print(
            f"{i}. {result['title']} "
            f"[{result['id']}] "
            f"history_score="
            f"{result['score']:.4f}"
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    checks = validate_results(
        results
    )

    print("\nVALIDATION")

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # -----------------------------------------------------
    # DYNAMIC HISTORY VERIFICATION
    #
    # Same current text but different historical pattern
    # -----------------------------------------------------

    comparison_history = (
        "Joy:0.82,"
        "Joy:0.78,"
        "Surprise:0.75,"
        "Joy:0.84"
    )

    comparison_records = parse_history(
        comparison_history
    )

    comparison_trend = analyze_history(
        comparison_records
    )

    comparison_results = generate_recommendations(
        comparison_trend
    )

    original_ranking = ranking_ids(
        results
    )

    changed_ranking = ranking_ids(
        comparison_results
    )

    print(
        "\nDYNAMIC HISTORY VERIFICATION"
    )

    print("-" * 45)

    print(
        "Original historical pattern : "
        + " > ".join(original_ranking)
    )

    print(
        "Changed historical pattern  : "
        + " > ".join(changed_ranking)
    )

    ranking_changed = (
        original_ranking != changed_ranking
    )

    print(
        "Recommendations changed due "
        "to emotional history: "
        + (
            "PASS"
            if ranking_changed
            else "FAIL"
        )
    )

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    all_checks_passed = (
        all(checks.values())
        and ranking_changed
        and len(records) >= 2
    )

    print("\n" + "=" * 60)

    if all_checks_passed:
        print(
            "TASK 6 OVERALL: "
            "ALL CHECKS PASSED"
        )
    else:
        print(
            "TASK 6 OVERALL: "
            "SOME CHECKS FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()