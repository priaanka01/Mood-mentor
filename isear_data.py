"""
Milestone 2 - ISEAR Dataset Preparation
=======================================

Loads and prepares the ISEAR emotion dataset for BERT/DistilBERT
training and held-out benchmark validation.

Responsibilities:
    1. Load ISEAR CSV files.
    2. Detect supported CSV formats automatically.
    3. Normalize ISEAR emotion labels.
    4. Add real Surprise examples when available.
    5. Create a reproducible stratified train/benchmark split.
    6. Support both:
       - 6-category project labels
       - 8-category native training labels
    7. Provide optional per-class row limiting.

The project output labels are:
    Joy, Sadness, Anger, Fear, Surprise, Disgust

ISEAR additionally contains:
    Shame, Guilt

For native 8-way training, Shame and Guilt remain separate.
For the public 6-category output, both are mapped to Sadness.

The benchmark split uses the project's 6-category schema and must
remain reproducible because Task 5/6 evaluation depends on the same
fraction and seed used during training.
"""

import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from emotion_labels import (
    EMOTION_LABELS,
    ISEAR_LABEL_MAP,
    ISEAR_LABEL_TYPOS,
    normalize_isear_label,
    normalize_isear_label_native,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_BENCHMARK_FRACTION = 0.15
DEFAULT_SEED = 42
DEFAULT_SURPRISE_CSV = "surprise_examples.csv"

TEXT_COLUMN_CANDIDATES = [
    "text",
    "sit",
    "situation",
    "sentence",
    "content",
]

LABEL_COLUMN_CANDIDATES = [
    "emotion",
    "field1",
    "label",
    "class",
    "sentiment",
]


# ---------------------------------------------------------------------
# Surprise fallback data
# ---------------------------------------------------------------------
#
# ISEAR has no native Surprise category. The project therefore uses
# surprise_examples.csv when available.
#
# These fallback examples are retained only so the pipeline can still
# run if the external Surprise file is unavailable.
# ---------------------------------------------------------------------

SURPRISE_SUPPLEMENT: List[str] = [
    "I couldn't believe it when they announced the promotion out of nowhere.",
    "The office threw me a surprise party and I had absolutely no idea.",
    "I was stunned when the project got approved a week ahead of schedule.",
    "Out of nowhere, my manager thanked me in front of the whole team.",
    "I never expected the client to say yes on the first call.",
    "The results came back completely different from what anyone predicted.",
    "Wait, what? I had no idea the merger was even being discussed.",
    "I opened my inbox and found out I'd been nominated for an award I never applied for.",
    "Nobody told me it was my last day working with this team until this morning.",
    "The new hire turned out to be someone I used to work with a decade ago.",
    "I was completely caught off guard when the CEO walked into our team meeting.",
    "Honestly, I gasped out loud when I saw the final budget numbers.",
    "They rearranged the entire org chart overnight and nobody saw it coming.",
    "I thought the deadline was next month, not tomorrow morning!",
    "My performance review came back so much better than I ever expected.",
    "The vendor called to say the shipment arrived three weeks early.",
    "I walked into the conference room and everyone was already there waiting for me.",
    "Turns out the intern had been quietly running the whole migration by herself.",
    "I found a handwritten thank-you note on my desk and had no idea who left it.",
    "The system that we thought was broken for months just started working again on its own.",
    "She got engaged last weekend and told absolutely nobody beforehand.",
    "I checked the leaderboard and somehow I was in first place.",
    "The weather completely flipped from sunny to a thunderstorm in ten minutes.",
    "Out of the blue, an old college friend showed up at my office.",
    "I didn't expect the interview to go that well, honestly.",
    "The test results were the opposite of what the doctor predicted.",
    "He walked in wearing a suit and I almost didn't recognize him.",
    "I had no clue the company was being acquired until the all-hands email.",
    "The score at halftime completely flipped by the end of the game.",
    "I opened the box and it wasn't at all what I ordered - it was better!",
]


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class EmotionRecord:
    """A text example and its normalized emotion label."""

    text: str
    label: str


@dataclass
class _RawRow:
    """Raw dataset row before project-label normalization."""

    text: str
    raw_label: str


class ISEARLoadError(Exception):
    """Raised when an ISEAR file cannot be parsed correctly."""


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------


def _prepend(first_row: List[str], rows: Iterable[List[str]]):
    """Return an iterator containing first_row followed by rows."""
    yield first_row
    yield from rows


def _detect_column(
    header: List[str],
    candidates: List[str],
) -> Optional[int]:
    """Find the first matching column name in a CSV header."""

    normalized_header = [
        column.strip().lower()
        for column in header
    ]

    for candidate in candidates:
        if candidate in normalized_header:
            return normalized_header.index(candidate)

    return None


def _is_recognized_label(cell: str) -> bool:
    """Check whether a CSV cell contains a known ISEAR label."""

    key = cell.strip().lower()

    return (
        key in ISEAR_LABEL_MAP
        or key in ISEAR_LABEL_TYPOS
    )


def _load_isear_csv_raw(filepath: str) -> List[_RawRow]:
    """
    Load an ISEAR CSV and return validated raw rows.

    Supported formats:

    1. Headered CSV:
       text,some_other_columns,label

    2. Headerless label-first CSV:
       joy,text...
       sadness,text...
       anger,text...

    Labels remain in their native form here. Shame and Guilt are
    therefore NOT merged at this stage.
    """

    if not os.path.isfile(filepath):
        raise ISEARLoadError(
            f"ISEAR file not found: '{filepath}'"
        )

    with open(
        filepath,
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.reader(file)

        try:
            first_row = next(reader)
        except StopIteration:
            raise ISEARLoadError(
                f"ISEAR file is empty: '{filepath}'"
            )

        text_index: Optional[int]
        label_index: Optional[int]

        # -------------------------------------------------------------
        # Detect headerless label-first format
        # -------------------------------------------------------------

        if (
            len(first_row) >= 2
            and _is_recognized_label(first_row[0])
        ):
            text_index = 1
            label_index = 0

            rows = _prepend(first_row, reader)

            print(
                "[isear_data] Detected headerless "
                "label-first CSV format."
            )

        # -------------------------------------------------------------
        # Detect headered format
        # -------------------------------------------------------------

        else:
            text_index = _detect_column(
                first_row,
                TEXT_COLUMN_CANDIDATES,
            )

            label_index = _detect_column(
                first_row,
                LABEL_COLUMN_CANDIDATES,
            )

            if (
                text_index is None
                or label_index is None
            ):
                raise ISEARLoadError(
                    "Could not detect text/label columns "
                    f"in header {first_row}. "
                    f"Expected text columns: "
                    f"{TEXT_COLUMN_CANDIDATES}; "
                    f"label columns: "
                    f"{LABEL_COLUMN_CANDIDATES}."
                )

            rows = reader

        # -------------------------------------------------------------
        # Read and validate rows
        # -------------------------------------------------------------

        raw_rows: List[_RawRow] = []
        skipped = 0

        for row in rows:

            if len(row) <= max(
                text_index,
                label_index,
            ):
                skipped += 1
                continue

            text = row[text_index].strip()
            raw_label = row[label_index].strip()

            if not text or not raw_label:
                skipped += 1
                continue

            normalized_label = raw_label.lower()
            normalized_label = ISEAR_LABEL_TYPOS.get(
                normalized_label,
                normalized_label,
            )

            if normalized_label not in ISEAR_LABEL_MAP:
                skipped += 1
                continue

            raw_rows.append(
                _RawRow(
                    text=text,
                    raw_label=normalized_label,
                )
            )

    print(
        f"[isear_data] Loaded {len(raw_rows)} valid rows "
        f"from '{filepath}' "
        f"({skipped} skipped: empty/unrecognized)."
    )

    return raw_rows


# ---------------------------------------------------------------------
# Public CSV loading
# ---------------------------------------------------------------------


def load_isear_csv(
    filepath: str,
) -> List[EmotionRecord]:
    """
    Load ISEAR data using the project's 6-category schema.

    Shame and Guilt are mapped to Sadness through
    normalize_isear_label().
    """

    raw_rows = _load_isear_csv_raw(filepath)

    return [
        EmotionRecord(
            text=row.text,
            label=normalize_isear_label(
                row.raw_label
            ),
        )
        for row in raw_rows
    ]


# ---------------------------------------------------------------------
# Surprise supplementation
# ---------------------------------------------------------------------


def _load_surprise_texts(
    surprise_csv_path: str,
) -> List[str]:
    """
    Load Surprise examples.

    Real examples from surprise_examples.csv are preferred.
    Fallback examples are used only when that file is unavailable.
    """

    if os.path.isfile(surprise_csv_path):

        with open(
            surprise_csv_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if not reader.fieldnames or "text" not in reader.fieldnames:
                raise ISEARLoadError(
                    f"Surprise file '{surprise_csv_path}' "
                    "must contain a 'text' column."
                )

            texts = [
                row["text"].strip()
                for row in reader
                if row.get("text", "").strip()
            ]

        print(
            f"[isear_data] Using {len(texts)} real "
            f"Surprise examples from "
            f"'{surprise_csv_path}'."
        )

        return texts

    print(
        f"[isear_data] '{surprise_csv_path}' not found - "
        f"using {len(SURPRISE_SUPPLEMENT)} fallback "
        "Surprise examples."
    )

    return SURPRISE_SUPPLEMENT.copy()


def _surprise_supplement_raw(
    surprise_csv_path: str = DEFAULT_SURPRISE_CSV,
) -> List[_RawRow]:
    """Return Surprise examples as raw dataset rows."""

    texts = _load_surprise_texts(
        surprise_csv_path
    )

    return [
        _RawRow(
            text=text,
            raw_label="surprise",
        )
        for text in texts
    ]


def with_surprise_supplement(
    records: List[EmotionRecord],
    surprise_csv_path: str = DEFAULT_SURPRISE_CSV,
) -> List[EmotionRecord]:
    """
    Add Surprise examples to an existing 6-category dataset.
    """

    supplemented = list(records)

    texts = _load_surprise_texts(
        surprise_csv_path
    )

    supplemented.extend(
        EmotionRecord(
            text=text,
            label="Surprise",
        )
        for text in texts
    )

    return supplemented


# ---------------------------------------------------------------------
# Train / benchmark splitting
# ---------------------------------------------------------------------


def train_benchmark_split(
    records: List[EmotionRecord],
    benchmark_fraction: float = DEFAULT_BENCHMARK_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    List[EmotionRecord],
    List[EmotionRecord],
]:
    """
    Create a reproducible stratified train/benchmark split.

    The benchmark contains benchmark_fraction of each class.
    """

    if not 0 < benchmark_fraction < 1:
        raise ValueError(
            "benchmark_fraction must be between 0 and 1."
        )

    rng = random.Random(seed)

    by_label: Dict[
        str,
        List[EmotionRecord],
    ] = {
        label: []
        for label in EMOTION_LABELS
    }

    for record in records:
        by_label.setdefault(
            record.label,
            [],
        ).append(record)

    train_records: List[EmotionRecord] = []
    benchmark_records: List[EmotionRecord] = []

    for label, items in by_label.items():

        shuffled_items = items[:]
        rng.shuffle(shuffled_items)

        benchmark_size = (
            max(
                1,
                int(
                    len(shuffled_items)
                    * benchmark_fraction
                ),
            )
            if shuffled_items
            else 0
        )

        benchmark_records.extend(
            shuffled_items[:benchmark_size]
        )

        train_records.extend(
            shuffled_items[benchmark_size:]
        )

    rng.shuffle(train_records)
    rng.shuffle(benchmark_records)

    return train_records, benchmark_records


# ---------------------------------------------------------------------
# Optional training-data size control
# ---------------------------------------------------------------------


def cap_rows_per_class(
    records: List[EmotionRecord],
    max_per_class: int,
    seed: int = DEFAULT_SEED,
) -> List[EmotionRecord]:
    """
    Limit the number of rows per class.

    This is useful for reducing training time while preserving
    representation from every class.
    """

    if max_per_class <= 0:
        raise ValueError(
            "max_per_class must be greater than 0."
        )

    rng = random.Random(seed)

    by_label: Dict[
        str,
        List[EmotionRecord],
    ] = defaultdict(list)

    for record in records:
        by_label[record.label].append(record)

    capped_records: List[EmotionRecord] = []

    for items in by_label.values():

        shuffled_items = items[:]
        rng.shuffle(shuffled_items)

        capped_records.extend(
            shuffled_items[:max_per_class]
        )

    rng.shuffle(capped_records)

    return capped_records


# ---------------------------------------------------------------------
# Standard 6-category dataset
# ---------------------------------------------------------------------


def prepare_isear_dataset(
    filepath: str,
    benchmark_fraction: float = DEFAULT_BENCHMARK_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    List[EmotionRecord],
    List[EmotionRecord],
]:
    """
    Prepare the standard 6-category ISEAR dataset.

    Returns:
        train_records
        benchmark_records
    """

    records = load_isear_csv(filepath)

    records = with_surprise_supplement(records)

    train_records, benchmark_records = train_benchmark_split(
        records,
        benchmark_fraction=benchmark_fraction,
        seed=seed,
    )

    train_counts = {
        label: sum(
            record.label == label
            for record in train_records
        )
        for label in EMOTION_LABELS
    }

    benchmark_counts = {
        label: sum(
            record.label == label
            for record in benchmark_records
        )
        for label in EMOTION_LABELS
    }

    print(
        f"[isear_data] Train: "
        f"{len(train_records)} rows "
        f"{train_counts}"
    )

    print(
        f"[isear_data] Benchmark "
        f"(held out, Task 6): "
        f"{len(benchmark_records)} rows "
        f"{benchmark_counts}"
    )

    return train_records, benchmark_records


# ---------------------------------------------------------------------
# Native 8-category training dataset
# ---------------------------------------------------------------------


def prepare_isear_training_dataset(
    filepath: str,
    benchmark_fraction: float = DEFAULT_BENCHMARK_FRACTION,
    seed: int = DEFAULT_SEED,
    surprise_csv_path: str = DEFAULT_SURPRISE_CSV,
) -> Tuple[
    List[EmotionRecord],
    List[EmotionRecord],
]:
    """
    Prepare data for native 8-category training.

    Training labels:
        Joy
        Sadness
        Anger
        Fear
        Surprise
        Disgust
        Shame
        Guilt

    Shame and Guilt remain separate during training.

    The benchmark is still returned using the project's standard
    6-category schema so Task 5/6 evaluation remains compatible.
    """

    raw_rows = _load_isear_csv_raw(filepath)

    raw_rows.extend(
        _surprise_supplement_raw(
            surprise_csv_path
        )
    )

    rng = random.Random(seed)

    # Group according to the public 6-category mapping so that the
    # benchmark split remains identical to prepare_isear_dataset().
    by_project_label: Dict[
        str,
        List[_RawRow],
    ] = {
        label: []
        for label in EMOTION_LABELS
    }

    for row in raw_rows:

        project_label = normalize_isear_label(
            row.raw_label
        )

        by_project_label[
            project_label
        ].append(row)

    train_raw: List[_RawRow] = []
    benchmark_raw: List[_RawRow] = []

    for items in by_project_label.values():

        shuffled_items = items[:]
        rng.shuffle(shuffled_items)

        benchmark_size = (
            max(
                1,
                int(
                    len(shuffled_items)
                    * benchmark_fraction
                ),
            )
            if shuffled_items
            else 0
        )

        benchmark_raw.extend(
            shuffled_items[:benchmark_size]
        )

        train_raw.extend(
            shuffled_items[benchmark_size:]
        )

    rng.shuffle(train_raw)
    rng.shuffle(benchmark_raw)

    # Native 8-way training labels.
    train_records = [
        EmotionRecord(
            text=row.text,
            label=normalize_isear_label_native(
                row.raw_label
            ),
        )
        for row in train_raw
    ]

    # Standard 6-category benchmark labels.
    benchmark_records = [
        EmotionRecord(
            text=row.text,
            label=normalize_isear_label(
                row.raw_label
            ),
        )
        for row in benchmark_raw
    ]

    train_counts = dict(
        defaultdict(int)
    )

    for record in train_records:
        train_counts[record.label] += 1

    benchmark_counts = dict(
        defaultdict(int)
    )

    for record in benchmark_records:
        benchmark_counts[record.label] += 1

    print(
        "[isear_data] Train "
        "(8-way native, Shame/Guilt kept distinct): "
        f"{len(train_records)} rows "
        f"{train_counts}"
    )

    print(
        "[isear_data] Benchmark "
        "(held out, 6-category, Task 6): "
        f"{len(benchmark_records)} rows "
        f"{benchmark_counts}"
    )

    return train_records, benchmark_records