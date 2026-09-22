"""
Mood Mentor - Milestone 3 Tasks 1-5 Integrated Validation

Runs the completed Milestone 3 flow:
1. Emotional state + intensity
2. Personalized recommendation
3. Hybrid recommendation
4. Dynamic ranking
5. Semantic wellness matching
"""

import argparse

from m3_emotion_state import analyze_emotional_state
from multi_label_classifier import MultiLabelBERTClassifier
from m3_task2_personalized_recommendation import load_content, recommend
from m3_task3_hybrid_recommendation import hybrid_recommend
from m3_task4_recommendation_ranking import rank_recommendations, validate_ranking
from m3_task5_semantic_wellness_matching import semantic_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", default="./multilabel_emotion_model")
    parser.add_argument("--content", default="wellness_content.csv")
    parser.add_argument("--preferences", default="breathing,mindfulness")
    parser.add_argument("--history", default="W03,W07")
    parser.add_argument("--trend", default="Fear,Fear,Sadness")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    classifier = MultiLabelBERTClassifier.load(args.model_dir)
    content = load_content(args.content)

    state = analyze_emotional_state(
        args.text, classifier, threshold=args.threshold
    )

    preferences = args.preferences.split(",") if args.preferences else []
    history = args.history.split(",") if args.history else []
    trend = args.trend.split(",") if args.trend else []

    personalized = recommend(
        state, content, preferences, history, trend, args.top_k
    )
    hybrid = hybrid_recommend(
        state, content, preferences, history, args.top_k
    )
    ranked = rank_recommendations(
        state, content, preferences, history, history, args.top_k
    )
    _, semantic = semantic_match(state, content)
    semantic = semantic[:args.top_k]

    print("\n" + "=" * 78)
    print("MOOD MENTOR - MILESTONE 3 TASKS 1-5 INTEGRATED VALIDATION")
    print("=" * 78)

    print("\n[1] EMOTIONAL STATE")
    print(f"Dominant emotion : {state.dominant_emotion}")
    print(f"Emotions         : {', '.join(state.multiple_emotions)}")
    print(f"Confidence       : {state.emotion_confidence:.4f}")
    print(f"Intensity        : {state.emotional_intensity:.4f}")
    print(f"Polarity         : {state.polarity}")
    print(f"Mixed state      : {state.mixed_emotional_state}")
    print(f"Severity         : {state.severity}")

    print("\n[2] PERSONALIZED TOP RECOMMENDATIONS")
    for i, x in enumerate(personalized, 1):
        print(f"{i}. {x['title']} -> {x['personalized_score']:.4f}")

    print("\n[3] HYBRID TOP RECOMMENDATIONS")
    for i, x in enumerate(hybrid, 1):
        print(f"{i}. {x['title']} -> {x['hybrid_score']:.4f}")

    print("\n[4] DYNAMIC RANKING")
    for i, x in enumerate(ranked, 1):
        print(f"{i}. {x['title']} -> {x['ranking_score']:.4f}")
    checks = validate_ranking(ranked)
    print("Ranking checks:", "PASS" if all(checks.values()) else "FAIL")

    print("\n[5] SEMANTIC MATCHING")
    for i, x in enumerate(semantic, 1):
        print(f"{i}. {x['title']} -> {x['semantic_similarity']:.4f}")

    print("\n" + "=" * 78)
    print("MILESTONE 3 TASKS 1-5: INTEGRATED FLOW COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()
