"""
Mood Mentor - Milestone 1 CLI
=================================
Run the full pipeline on a text input, .txt file, or .csv file and
save a report — without writing any Python each time.

Usage:
    python3 main.py --text "I love this!"
    python3 main.py --txt path/to/file.txt
    python3 main.py --csv sample_corpus.csv
    python3 main.py --csv sample_corpus.csv --column review
    python3 main.py --csv sample_corpus.csv --out my_report

    --out sets the output filename prefix (default: "report").
    Produces <out>.csv and <out>.json in the current folder.
"""

import argparse
import sys

from pipeline import (
    run_pipeline_from_text,
    run_pipeline_from_txt_file,
    run_pipeline_from_csv_file,
)
from report_generator import build_report, save_report_csv, save_report_json
from text_ingestion import IngestionError


def main():
    parser = argparse.ArgumentParser(description="Mood Mentor Milestone 1 pipeline runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Analyze a single string of text")
    group.add_argument("--txt", help="Path to a .txt file to analyze")
    group.add_argument("--csv", help="Path to a .csv file to analyze")
    parser.add_argument("--column", default=None, help="Column name to read text from (for --csv)")
    parser.add_argument("--out", default="report", help="Output filename prefix (default: report)")
    args = parser.parse_args()

    try:
        if args.text:
            result = run_pipeline_from_text(args.text)
            rows = [result.as_dict()]
        elif args.txt:
            result = run_pipeline_from_txt_file(args.txt)
            rows = [result.as_dict()]
        else:
            results = run_pipeline_from_csv_file(args.csv, text_column=args.column)
            rows = [r.as_dict() for r in results]
    except IngestionError as e:
        print(f"Error: {e}")
        sys.exit(1)

    report = build_report(rows)

    csv_path = f"{args.out}.csv"
    json_path = f"{args.out}.json"
    save_report_csv(report, csv_path)
    save_report_json(report, json_path)

    print(f"\nAnalyzed {report['summary']['num_samples_analyzed']} sample(s)")
    print(f"  Positive: {report['summary']['positive_count']}")
    print(f"  Negative: {report['summary']['negative_count']}")
    print(f"  Neutral:  {report['summary']['neutral_count']}")
    print(f"\nSaved: {csv_path}, {json_path}")

    for r in rows:
        text_preview = r["input_text"][:60]
        print(f"  {r['sentiment_label']:9s} | compound={r['compound_score']:+.3f} | {text_preview}")


if __name__ == "__main__":
    main()