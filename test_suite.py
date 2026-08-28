"""
Milestone 1 Validation Test Suite
=====================================
Runs assertions for every checklist item in Tasks 1-5 of the
"Text Ingestion & Baseline Sentiment" milestone, and prints a
pass/fail report. Exits non-zero if anything fails, so it can be
wired into CI.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from text_ingestion import (
    ingest_text_input, ingest_txt_file, ingest_csv_file,
    validate_input, IngestionError,
)
from preprocessing import preprocess, remove_noise
from sentiment_analysis import analyze_sentiment
from report_generator import analyze_corpus, save_report_csv, save_report_json
from pipeline import run_pipeline_from_text, run_pipeline_from_txt_file, run_pipeline_from_csv_file

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------
print("\n=== Task 1: Text Ingestion Workflow ===")

# direct text input
rec = ingest_text_input("This product is fantastic!")
check("T1.1 direct text input ingested", rec.raw_text == "This product is fantastic!")

# .txt upload
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("Sample content from a text file upload.")
    txt_path = f.name
rec_txt = ingest_txt_file(txt_path)
check("T1.2 .txt file ingested", rec_txt.raw_text.startswith("Sample content"))
os.unlink(txt_path)

# .csv upload
csv_records = ingest_csv_file(os.path.join(os.path.dirname(__file__), "sample_corpus.csv"))
check("T1.3 .csv file ingested (multi-row)", len(csv_records) >= 8, f"got {len(csv_records)} rows")

# read input data (already exercised above); validate format
check("T1.4 validate_input accepts normal text", validate_input("hello world") == "hello world")

# invalid format: non-string
try:
    validate_input(12345)
    check("T1.5 rejects non-string input", False)
except IngestionError:
    check("T1.5 rejects non-string input", True)

# empty / whitespace-only input handling
for bad in ["", "   ", None, "!!!   ***"]:
    try:
        validate_input(bad)
        check(f"T1.6 rejects invalid input {bad!r}", False)
    except IngestionError:
        check(f"T1.6 rejects invalid input {bad!r}", True)

# valid text passed cleanly to preprocessing
prep_check = preprocess(validate_input("Valid text ready for preprocessing."))
check("T1.7 valid text flows into preprocessing", prep_check.processed_text != "")


# ---------------------------------------------------------------
print("\n=== Task 2: Preprocessing Validation ===")

p1 = preprocess("The quick brown foxes are jumping over the lazy dogs!!")
check("T2.1 tokenization occurs", len(p1.tokens) > 0)
check("T2.2 stop-word removal (no 'the'/'are')", "the" not in p1.tokens and "are" not in p1.tokens)
check("T2.3 lemmatization ('foxes'->'fox' or 'jumping'->'jump')",
      "fox" in p1.tokens or "jump" in p1.tokens or "dog" in p1.tokens,
      detail=str(p1.tokens))
check("T2.4 punctuation stripped from tokens", "!" not in p1.tokens and "!!" not in p1.tokens)

p_noise = remove_noise("Check this out http://example.com <b>bold</b>   text!! 😀😀")
check("T2.5 URL removed from noise filtering", "http://example.com" not in p_noise)
check("T2.6 HTML tags removed", "<b>" not in p_noise)
check("T2.7 repeated spaces collapsed", "   " not in p_noise)

p_special = preprocess("Wow!!! @#$% This... is *amazing* — really??")
check("T2.8 special characters handled without crash", p_special.processed_text != "" )

p_empty = preprocess("")
check("T2.9 empty text handled gracefully (no crash, empty output)",
      p_empty.tokens == [] and p_empty.processed_text == "")

p_spaces = preprocess("word1     word2        word3")
check("T2.10 repeated-space input tokenizes correctly", len(p_spaces.tokens) == 3, detail=str(p_spaces.tokens))

short = preprocess("Good.")
long_text = preprocess(" ".join(["This is a moderately long piece of text meant to test the pipeline."] * 20))
check("T2.11 handles very short text", short.processed_text != "")
check("T2.12 handles long text", len(long_text.tokens) > 20)


# ---------------------------------------------------------------
print("\n=== Task 3: VADER Sentiment Validation ===")

pos = analyze_sentiment("I absolutely love this, it's amazing and wonderful!")
neg = analyze_sentiment("This is terrible, I hate it, worst experience ever.")
neu = analyze_sentiment("The report is due on Thursday at noon.")

check("T3.1 positive sentiment detected", pos.label == "Positive", detail=str(pos))
check("T3.2 negative sentiment detected", neg.label == "Negative", detail=str(neg))
check("T3.3 neutral sentiment detected", neu.label == "Neutral", detail=str(neu))

check("T3.4 compound score present & in range", -1.0 <= pos.compound <= 1.0)
check("T3.5 positive score present & in range", 0.0 <= pos.pos <= 1.0)
check("T3.6 negative score present & in range", 0.0 <= pos.neg <= 1.0)
check("T3.7 neutral score present & in range", 0.0 <= pos.neu <= 1.0)

# scores must differ across different inputs -> proves nothing is hardcoded
check("T3.8 scores are NOT hardcoded (differ across distinct inputs)",
      pos.compound != neg.compound and pos.compound != neu.compound and neg.compound != neu.compound,
      detail=f"pos={pos.compound} neg={neg.compound} neu={neu.compound}")

# pos+neg+neu should sum to ~1.0 (VADER's own invariant, confirms genuine computation)
total = round(pos.pos + pos.neg + pos.neu, 2)
check("T3.9 pos+neg+neu sums to ~1.0 (genuine VADER output)", 0.98 <= total <= 1.02, detail=str(total))


# ---------------------------------------------------------------
print("\n=== Task 4: Initial Sentiment Report Validation ===")

samples = [
    "I love this so much, best day ever!",
    "I hate waiting in long lines, so annoying.",
    "The train departs at 9am.",
    "",  # invalid, should be skipped
    "   ",  # invalid, should be skipped
    "Absolutely fantastic service, will come back again!",
]
report = analyze_corpus(samples)
check("T4.1 report generated with rows", len(report["rows"]) > 0)
check("T4.2 invalid samples skipped, not analyzed", report["summary"]["num_samples_skipped_invalid"] == 2)
check("T4.3 num_samples_analyzed correct", report["summary"]["num_samples_analyzed"] == 4)
check("T4.4 each row has input_text and processed_text",
      all("input_text" in r and "processed_text" in r for r in report["rows"]))
check("T4.5 each row has sentiment classification + scores",
      all(r["sentiment_label"] in ("Positive", "Negative", "Neutral") for r in report["rows"]))
check("T4.6 pos/neg/neu counts sum to total analyzed",
      report["summary"]["positive_count"] + report["summary"]["negative_count"] + report["summary"]["neutral_count"]
      == report["summary"]["num_samples_analyzed"])

out_dir = tempfile.mkdtemp()
save_report_csv(report, os.path.join(out_dir, "report.csv"))
save_report_json(report, os.path.join(out_dir, "report.json"))
check("T4.7 report saved to CSV", os.path.exists(os.path.join(out_dir, "report.csv")))
check("T4.8 report saved to JSON", os.path.exists(os.path.join(out_dir, "report.json")))


# ---------------------------------------------------------------
print("\n=== Task 5: Complete Pipeline Integration ===")

r_text = run_pipeline_from_text("This is wonderful news, I'm so happy!")
check("T5.1 text-input -> preprocessing -> sentiment integration works",
      r_text.sent.label == "Positive" and r_text.prep.processed_text != "")

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("This is a disaster, everything went wrong today.")
    txt_path2 = f.name
r_txtfile = run_pipeline_from_txt_file(txt_path2)
check("T5.2 txt-file -> preprocessing -> sentiment integration works",
      r_txtfile.sent.label == "Negative")
os.unlink(txt_path2)

r_csv = run_pipeline_from_csv_file(os.path.join(os.path.dirname(__file__), "sample_corpus.csv"))
check("T5.3 csv-file -> preprocessing -> sentiment integration works (multi-row)",
      len(r_csv) >= 8 and all(r.sent.label in ("Positive", "Negative", "Neutral") for r in r_csv))
check("T5.4 module hand-off preserves data (raw_text == prep.original_text)",
      all(r.record.raw_text == r.prep.original_text for r in r_csv))
check("T5.5 every pipeline result serializes cleanly (as_dict)",
      all(set(r.as_dict().keys()) == {
          "source", "origin", "input_text", "processed_text", "sentiment_label",
          "compound_score", "positive_score", "negative_score", "neutral_score"
      } for r in r_csv))


# ---------------------------------------------------------------
print(f"\n=== SUMMARY: {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    print("Failed checks:", FAIL)
    sys.exit(1)
else:
    print("All Milestone 1 checks passed.")
    sys.exit(0)
