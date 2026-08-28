"""
Task 5 - Complete Pipeline Integration
==========================================
Wires together Tasks 1-4 into one callable pipeline, and offers
entry points for each of the three supported input methods
(direct text, .txt file, .csv file), so the same integration path
is exercised regardless of how data enters the system.
"""

from typing import List, Dict, Any

from text_ingestion import (
    ingest_text_input,
    ingest_txt_file,
    ingest_csv_file,
    IngestedRecord,
    IngestionError,
)
from preprocessing import preprocess, PreprocessResult
from sentiment_analysis import analyze_sentiment, SentimentResult


class PipelineResult:
    def __init__(self, record: IngestedRecord, prep: PreprocessResult, sent: SentimentResult):
        self.record = record
        self.prep = prep
        self.sent = sent

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.record.source,
            "origin": self.record.origin,
            "input_text": self.prep.original_text,
            "processed_text": self.prep.processed_text,
            "sentiment_label": self.sent.label,
            "compound_score": self.sent.compound,
            "positive_score": self.sent.pos,
            "negative_score": self.sent.neg,
            "neutral_score": self.sent.neu,
        }


def run_pipeline_on_record(record: IngestedRecord) -> PipelineResult:
    """Module hand-off: ingestion -> preprocessing -> sentiment."""
    prep = preprocess(record.raw_text)
    sent = analyze_sentiment(prep.cleaned_text)
    return PipelineResult(record, prep, sent)


def run_pipeline_from_text(text: str) -> PipelineResult:
    record = ingest_text_input(text)
    return run_pipeline_on_record(record)


def run_pipeline_from_txt_file(filepath: str) -> PipelineResult:
    record = ingest_txt_file(filepath)
    return run_pipeline_on_record(record)


def run_pipeline_from_csv_file(filepath: str, text_column: str = None) -> List[PipelineResult]:
    records = ingest_csv_file(filepath, text_column=text_column)
    return [run_pipeline_on_record(r) for r in records]
