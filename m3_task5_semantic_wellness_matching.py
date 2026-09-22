"""
Mood Mentor - Milestone 3 Task 5
Semantic Wellness Content Matching.

Flow:
User emotional state -> sentence embedding ->
wellness content embeddings -> cosine similarity -> ranked content.

Dependency:
    pip install sentence-transformers

The embedding model is downloaded once by Hugging Face and then cached.
"""

import argparse
import csv
from typing import List

from sentence_transformers import SentenceTransformer, util

from m3_emotion_state import analyze_emotional_state
from multi_label_classifier import MultiLabelBERTClassifier


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_content(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_content_text(item: dict) -> str:
    return (
        f"{item['title']}. {item['description']}. "
        f"Useful for emotions: {item['emotions']}. "
        f"Tags: {item['tags']}."
    )


def build_emotional_query(state) -> str:
    emotions = ", ".join(state.multiple_emotions)
    return (
        f"The user is feeling {emotions}. "
        f"The dominant emotion is {state.dominant_emotion}. "
        f"Emotional intensity is {state.emotional_intensity:.2f}. "
        f"Polarity is {state.polarity}. "
        f"The emotional severity is {state.severity}."
    )


def semantic_match(state, content, model_name=DEFAULT_EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)

    query = build_emotional_query(state)
    content_texts = [build_content_text(x) for x in content]

    query_embedding = model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )
    content_embeddings = model.encode(
        content_texts, convert_to_tensor=True, normalize_embeddings=True
    )

    similarities = util.cos_sim(query_embedding, content_embeddings)[0]

    results = []
    for item, score in zip(content, similarities):
        row = dict(item)
        row["semantic_similarity"] = round(float(score), 4)
        results.append(row)

    results.sort(
        key=lambda x: (-x["semantic_similarity"], x["content_id"])
    )
    return query, results


def validate_results(results):
    scores = [x["semantic_similarity"] for x in results]
    ids = [x["content_id"] for x in results]
    return {
        "similarity_in_range": all(-1.0 <= x <= 1.0 for x in scores),
        "ranking_order": scores == sorted(scores, reverse=True),
        "no_duplicates": len(ids) == len(set(ids)),
        "results_present": bool(results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", default="./multilabel_emotion_model")
    parser.add_argument("--content", default="wellness_content.csv")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    classifier = MultiLabelBERTClassifier.load(args.model_dir)
    state = analyze_emotional_state(
        args.text, classifier, threshold=args.threshold
    )

    content = load_content(args.content)
    query, results = semantic_match(
        state, content, model_name=args.embedding_model
    )
    results = results[:args.top_k]
    checks = validate_results(results)

    print("\n" + "=" * 70)
    print("TASK 5 - SEMANTIC WELLNESS CONTENT MATCHING")
    print("=" * 70)
    print(f"Semantic query: {query}")

    print("\nRanked wellness content:")
    for i, item in enumerate(results, 1):
        print(
            f"{i}. {item['title']} "
            f"[{item['content_id']}] "
            f"similarity={item['semantic_similarity']:.4f}"
        )

    print("\nValidation:")
    for name, passed in checks.items():
        print(
            f"  {name.replace('_', ' ').title():25s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    all_passed = all(checks.values())
    print(
        f"\nTASK 5 OVERALL: "
        f"{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}"
    )
    print("=" * 70)
