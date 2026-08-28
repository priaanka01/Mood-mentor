"""
Task 2 - Preprocessing Module
================================
Text preprocessing pipeline:
  1. Noise filtering (URLs, HTML tags, emojis/non-ascii noise, extra whitespace)
  2. Punctuation & special character removal
  3. Tokenization
  4. Stop-word removal
  5. Lemmatization

Design note: sentiment analysis (Task 3 / VADER) works BEST on text
that still has punctuation and case, because VADER's lexicon uses
punctuation emphasis (e.g. "!!!") and capitalization as intensity
signals. So this module returns BOTH:
  - `cleaned_text`   : lightly cleaned, case/punctuation preserved -> feed to VADER
  - `tokens`         : fully processed tokens (lowercased, no stopwords,
                        no punctuation, lemmatized) -> feed to any
                        token-based downstream analysis / reporting
"""

import re
import string
from dataclasses import dataclass, field
from typing import List

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ASCII_NOISE_RE = re.compile(r"[^\x00-\x7F]+")  # strips emojis/odd unicode noise


@dataclass
class PreprocessResult:
    original_text: str
    cleaned_text: str          # for VADER (case + punctuation intact)
    tokens: List[str] = field(default_factory=list)  # fully processed tokens
    processed_text: str = ""   # tokens joined back for reporting


def remove_noise(text: str) -> str:
    """Strip URLs, HTML tags, non-ASCII noise, and collapse repeated whitespace."""
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ASCII_NOISE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words using NLTK's word_tokenize."""
    return word_tokenize(text)


def remove_punctuation(tokens: List[str]) -> List[str]:
    """Drop tokens that are pure punctuation."""
    return [t for t in tokens if t not in string.punctuation and t.strip() != ""]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stop-words (case-insensitive)."""
    return [t for t in tokens if t.lower() not in _stopwords]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens to their base dictionary form (lowercased)."""
    return [_lemmatizer.lemmatize(t.lower()) for t in tokens]


def preprocess(text: str) -> PreprocessResult:
    """
    Full Task 2 pipeline.

    Handles edge cases explicitly:
      - Empty text            -> returns empty tokens/cleaned_text, no crash
      - Repeated spaces        -> collapsed in remove_noise
      - Special characters      -> stripped in remove_noise
      - Punctuation             -> stripped in remove_punctuation
      - Very short / long text  -> no special-casing needed, pipeline is length-agnostic
    """
    if text is None:
        text = ""

    original_text = text

    # Step 1: noise filtering (keep case + punctuation for VADER)
    cleaned_text = remove_noise(text)

    if cleaned_text == "":
        return PreprocessResult(
            original_text=original_text,
            cleaned_text="",
            tokens=[],
            processed_text="",
        )

    # Step 2-5: tokenize -> strip punctuation -> remove stopwords -> lemmatize
    raw_tokens = tokenize(cleaned_text)
    no_punct = remove_punctuation(raw_tokens)
    no_stop = remove_stopwords(no_punct)
    lemmas = lemmatize(no_stop)

    return PreprocessResult(
        original_text=original_text,
        cleaned_text=cleaned_text,
        tokens=lemmas,
        processed_text=" ".join(lemmas),
    )
