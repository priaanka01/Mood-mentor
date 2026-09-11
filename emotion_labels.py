"""
Milestone 2 - Shared Emotion Label Schema
==========================================

Defines the common emotion-label mapping used throughout the project.

Project emotion categories:
    Joy, Sadness, Anger, Fear, Surprise, Disgust

The ISEAR dataset contains:
    joy, fear, anger, sadness, disgust, shame, guilt

Because the project requires six categories:
    - Shame and Guilt are mapped to Sadness.
    - Surprise is supplemented from an additional labeled dataset.

The module also provides an internal 8-label training schema. Shame and
Guilt can be learned separately during model training and merged into
Sadness after prediction.
"""

from typing import Dict, List


# ---------------------------------------------------------------------------
# Project emotion labels
# ---------------------------------------------------------------------------

EMOTION_LABELS: List[str] = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
]

EMOTION_TO_INDEX: Dict[str, int] = {
    label: index
    for index, label in enumerate(EMOTION_LABELS)
}

INDEX_TO_EMOTION: Dict[int, str] = {
    index: label
    for label, index in EMOTION_TO_INDEX.items()
}


# ---------------------------------------------------------------------------
# ISEAR label normalization
# ---------------------------------------------------------------------------

# Maps ISEAR labels to the six project categories.
ISEAR_LABEL_MAP: Dict[str, str] = {
    "joy": "Joy",
    "fear": "Fear",
    "anger": "Anger",
    "sadness": "Sadness",
    "disgust": "Disgust",
    "shame": "Sadness",
    "guilt": "Sadness",
    "surprise": "Surprise",
}


# Known label typo found in some ISEAR CSV files.
ISEAR_LABEL_TYPOS: Dict[str, str] = {
    "guit": "guilt",
}


def normalize_isear_label(raw_label: str) -> str:
    """
    Convert a raw ISEAR label into one of the six project labels.

    Args:
        raw_label: Original emotion label from the dataset.

    Returns:
        One of the six project emotion labels.

    Raises:
        ValueError: If the label is not recognized.
    """
    key = raw_label.strip().lower()
    key = ISEAR_LABEL_TYPOS.get(key, key)

    if key not in ISEAR_LABEL_MAP:
        raise ValueError(
            f"Unrecognized ISEAR label '{raw_label}'. "
            f"Expected one of: {sorted(ISEAR_LABEL_MAP.keys())}"
        )

    return ISEAR_LABEL_MAP[key]


# ---------------------------------------------------------------------------
# Internal training-label schema
# ---------------------------------------------------------------------------

# Eight labels are used internally during training so that Shame and
# Guilt can be learned separately instead of being mixed directly into
# Sadness during training.
TRAINING_LABELS: List[str] = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
    "Shame",
    "Guilt",
]

TRAINING_LABEL_TO_INDEX: Dict[str, int] = {
    label: index
    for index, label in enumerate(TRAINING_LABELS)
}

INDEX_TO_TRAINING_LABEL: Dict[int, str] = {
    index: label
    for label, index in TRAINING_LABEL_TO_INDEX.items()
}


# Maps each internal training label to the final project label.
TRAINING_TO_PROJECT_LABEL: Dict[str, str] = {
    "Joy": "Joy",
    "Sadness": "Sadness",
    "Anger": "Anger",
    "Fear": "Fear",
    "Surprise": "Surprise",
    "Disgust": "Disgust",
    "Shame": "Sadness",
    "Guilt": "Sadness",
}


# Raw ISEAR labels mapped to the eight internal training labels.
_NATIVE_LABEL_MAP: Dict[str, str] = {
    "joy": "Joy",
    "fear": "Fear",
    "anger": "Anger",
    "sadness": "Sadness",
    "disgust": "Disgust",
    "shame": "Shame",
    "guilt": "Guilt",
    "surprise": "Surprise",
}


def normalize_isear_label_native(raw_label: str) -> str:
    """
    Normalize an ISEAR label without merging Shame and Guilt.

    This function is used when preparing the internal training data.

    Args:
        raw_label: Original ISEAR emotion label.

    Returns:
        One of the eight internal training labels.

    Raises:
        ValueError: If the label is not recognized.
    """
    key = raw_label.strip().lower()
    key = ISEAR_LABEL_TYPOS.get(key, key)

    if key not in _NATIVE_LABEL_MAP:
        raise ValueError(
            f"Unrecognized ISEAR label '{raw_label}'. "
            f"Expected one of: {sorted(_NATIVE_LABEL_MAP.keys())}"
        )

    return _NATIVE_LABEL_MAP[key]


# ---------------------------------------------------------------------------
# Probability merging
# ---------------------------------------------------------------------------

def merge_training_probs_softmax(
    training_probs: Dict[str, float],
) -> Dict[str, float]:
    """
    Merge eight-way softmax probabilities into the six project labels.

    Softmax outputs represent mutually exclusive classes, so probabilities
    belonging to the same project category can be summed.

    Shame + Guilt -> Sadness.
    """
    merged = {
        label: 0.0
        for label in EMOTION_LABELS
    }

    for training_label, probability in training_probs.items():
        project_label = TRAINING_TO_PROJECT_LABEL[training_label]
        merged[project_label] += probability

    return merged


def merge_training_probs_sigmoid(
    training_probs: Dict[str, float],
) -> Dict[str, float]:
    """
    Merge eight independent sigmoid probabilities into six project labels.

    Sigmoid outputs are independent probabilities and therefore cannot be
    safely summed. A noisy-OR calculation is used instead:

        merged = 1 - product(1 - probability)

    This combines multiple independent probabilities mapped to the same
    project emotion while keeping the final value between 0 and 1.
    """
    none_present = {
        label: 1.0
        for label in EMOTION_LABELS
    }

    for training_label, probability in training_probs.items():
        project_label = TRAINING_TO_PROJECT_LABEL[training_label]
        none_present[project_label] *= (1.0 - probability)

    return {
        label: round(1.0 - none_present[label], 4)
        for label in EMOTION_LABELS
    }