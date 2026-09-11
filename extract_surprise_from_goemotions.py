"""
Extract Real Surprise Examples from GoEmotions
===================================================
Replaces the hand-written SURPRISE_SUPPLEMENT placeholder in
isear_data.py with real, human-annotated text. GoEmotions is a 58k
Reddit-comment dataset with 27 emotion labels (Google Research),
downloadable directly - no huggingface.co account/gating needed.

Run this ONCE on your machine (needs internet - not this dev sandbox):

    python3 extract_surprise_from_goemotions.py

It will:
  1. Download the 3 raw GoEmotions CSVs (~50MB total) if not already present
  2. Keep only rows labeled Surprise AND no other emotion (clean single-
     label signal - a comment tagged both "surprise" and "joy" is a
     mixed-emotion example, not a good one to isolate what pure
     Surprise looks like for a 6-way single-label fine-tune)
  3. Drop rows GoEmotions' own raters flagged as "very unclear"
  4. De-duplicate (multiple raters can annotate the same comment)
  5. Save up to --max-examples rows to surprise_examples.csv, in the
     header format isear_data.py already knows how to read (text,label)

Then isear_data.py's with_surprise_supplement() will automatically use
this file instead of the small hardcoded placeholder list, if present.
"""

import argparse
import csv
import os
import urllib.request

GOEMOTIONS_URLS = [
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
]

# All 27 GoEmotions emotion columns + neutral. Used to check that a
# row is a CLEAN single-label Surprise example (surprise=1, every
# other one of these =0).
ALL_GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def download_if_missing(url: str, dest_dir: str) -> str:
    filename = os.path.join(dest_dir, os.path.basename(url))
    if os.path.exists(filename):
        print(f"[extract] Already downloaded: {filename}")
        return filename
    print(f"[extract] Downloading {url} ...")
    urllib.request.urlretrieve(url, filename)
    print(f"[extract] Saved to {filename}")
    return filename


def extract_clean_surprise_rows(csv_path: str) -> list:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("example_very_unclear", "").strip().upper() == "TRUE":
                continue
            try:
                is_surprise = int(row["surprise"]) == 1
                other_labels_are_zero = all(
                    int(row[label]) == 0 for label in ALL_GOEMOTIONS_LABELS if label != "surprise"
                )
            except (KeyError, ValueError):
                continue
            if is_surprise and other_labels_are_zero:
                text = row["text"].strip()
                if text:
                    rows.append(text)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract clean Surprise examples from GoEmotions")
    parser.add_argument("--data-dir", default="./goemotions_data", help="Where to download/cache the raw CSVs")
    parser.add_argument("--max-examples", type=int, default=300, help="Cap on output rows (default 300, matching the other emotion classes' per-class cap)")
    parser.add_argument("--output", default="surprise_examples.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    all_texts = []
    for url in GOEMOTIONS_URLS:
        path = download_if_missing(url, args.data_dir)
        texts = extract_clean_surprise_rows(path)
        print(f"[extract] {os.path.basename(path)}: {len(texts)} clean single-label Surprise rows")
        all_texts.extend(texts)

    # De-duplicate (multiple raters can annotate the same comment).
    seen = set()
    unique_texts = []
    for t in all_texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    print(f"[extract] {len(unique_texts)} unique clean Surprise examples total")

    import random
    random.Random(42).shuffle(unique_texts)
    selected = unique_texts[: args.max_examples]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text in selected:
            writer.writerow([text, "surprise"])

    print(f"[extract] Wrote {len(selected)} rows to '{args.output}'")
    print(f"[extract] Move/copy this file next to isear_data.py - it will be picked up automatically.")


if __name__ == "__main__":
    main()