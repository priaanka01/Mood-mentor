"""
Mood Mentor - Milestone 1 Entry Point
=========================================
Single consolidated entry point (previously split across main.py's
one-shot CLI and this file's interactive session — merged here to
remove the duplicated save/print/report logic between them).

Two modes, one script:

  1. INTERACTIVE (default, no arguments): Task 1's flow explicitly
     calls for enter text -> upload .txt -> upload .csv -> validate ->
     preprocess, as a live workflow rather than a one-shot CLI arg.
     This runs a menu loop: feed it text/files as you go, invalid
     input is caught and you're re-prompted (never crashes the
     session), and every valid entry accumulates into one running
     session. Generate ONE consolidated report on demand.

  2. ONE-SHOT (pass --text / --txt / --csv): scripted/automatable run
     for a single input — analyzes it and writes the report
     immediately, no menu. Useful for CI or quick one-off checks.

Usage:
    python3 milestone1.py                          # interactive session
    python3 milestone1.py --text "I love this!"    # one-shot
    python3 milestone1.py --txt path/to/file.txt
    python3 milestone1.py --csv sample_corpus.csv --column review --out my_report
"""

import argparse
import os
import sys

from text_ingestion import (
    ingest_text_input, ingest_txt_file, ingest_csv_file, IngestionError,
)
from pipeline import (
    run_pipeline_on_record,
    run_pipeline_from_text,
    run_pipeline_from_txt_file,
    run_pipeline_from_csv_file,
)
from report_generator import build_report, save_report, print_report_summary


def print_menu():
    print("\n" + "=" * 55)
    print(" Mood Mentor - Milestone 1 Interactive Session")
    print("=" * 55)
    print(f" Entries collected so far: {len(SESSION_ROWS)}")
    print("-" * 55)
    print(" 1) Enter text directly")
    print(" 2) Upload a .txt file")
    print(" 3) Upload a .csv file")
    print(" 4) Generate & download Milestone 1 report")
    print(" 5) Exit without generating a report")
    print("-" * 55)


def handle_direct_text():
    """Task 1: Create/enter text input, with an invalid-input retry loop."""
    text = input("\nType or paste your text (blank to cancel): ")
    if text.strip() == "":
        print("Cancelled — no text entered.")
        return
    try:
        record = ingest_text_input(text)
    except IngestionError as e:
        print(f"[Invalid input] {e}  Nothing was added — try again.")
        return

    result = run_pipeline_on_record(record)
    row = result.as_dict()
    SESSION_ROWS.append(row)
    print(f"-> Analyzed as {row['sentiment_label']} (compound={row['compound_score']:+.3f})")


def handle_txt_upload():
    """Task 1: Upload .txt file, validated and error-handled."""
    path = input("\nPath to .txt file (blank to cancel): ").strip().strip('"')
    if path == "":
        print("Cancelled — no path entered.")
        return
    try:
        record = ingest_txt_file(path)
    except IngestionError as e:
        print(f"[Invalid input] {e}  Nothing was added — try again.")
        return

    result = run_pipeline_on_record(record)
    row = result.as_dict()
    SESSION_ROWS.append(row)
    print(f"-> Loaded '{os.path.basename(path)}', analyzed as {row['sentiment_label']} "
          f"(compound={row['compound_score']:+.3f})")


def handle_csv_upload():
    """Task 1: Upload .csv file, validated row-by-row, invalid rows skipped not crashed on."""
    path = input("\nPath to .csv file (blank to cancel): ").strip().strip('"')
    if path == "":
        print("Cancelled — no path entered.")
        return
    column = input("Column name to read text from (blank = auto-detect): ").strip()
    column = column if column else None

    try:
        records = ingest_csv_file(path, text_column=column)
    except IngestionError as e:
        print(f"[Invalid input] {e}  Nothing was added — try again.")
        return

    added = 0
    for record in records:
        result = run_pipeline_on_record(record)
        SESSION_ROWS.append(result.as_dict())
        added += 1

    print(f"-> Loaded '{os.path.basename(path)}': {added} valid row(s) analyzed and added to session.")


def handle_generate_report():
    """Task 4/5: One consolidated, downloadable report for everything analyzed this session."""
    if not SESSION_ROWS:
        print("\nNo entries collected yet — add some text first (options 1-3).")
        return

    report = build_report(SESSION_ROWS)
    out_name = input("\nOutput filename (no extension, default 'milestone1_report'): ").strip()
    out_name = out_name if out_name else "milestone1_report"

    csv_path, json_path = save_report(report, out_name)
    print_report_summary(report, csv_path, json_path, header="MILESTONE 1 REPORT GENERATED")
    print(" Open the .csv in Excel, or download it from your file")
    print(" explorer at the path above — that's your Milestone 1")
    print(" deliverable.")


SESSION_ROWS = []


def run_interactive():
    while True:
        print_menu()
        choice = input(" Choose an option (1-5): ").strip()

        if choice == "1":
            handle_direct_text()
        elif choice == "2":
            handle_txt_upload()
        elif choice == "3":
            handle_csv_upload()
        elif choice == "4":
            handle_generate_report()
        elif choice == "5":
            print("\nExiting without generating a report. Bye!")
            sys.exit(0)
        else:
            print("\n[Invalid menu choice] Please enter a number from 1-5.")


def run_one_shot(args):
    """Former main.py behavior: single scripted run, report written immediately."""
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
    csv_path, json_path = save_report(report, args.out)
    print_report_summary(report, csv_path, json_path, header="MOOD MENTOR - ONE-SHOT RUN")

    for r in rows:
        text_preview = r["input_text"][:60]
        print(f"  {r['sentiment_label']:9s} | compound={r['compound_score']:+.3f} | {text_preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Mood Mentor Milestone 1 — interactive session by default, "
                     "or pass --text/--txt/--csv for a scripted one-shot run."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", help="Analyze a single string of text (one-shot mode)")
    group.add_argument("--txt", help="Path to a .txt file to analyze (one-shot mode)")
    group.add_argument("--csv", help="Path to a .csv file to analyze (one-shot mode)")
    parser.add_argument("--column", default=None, help="Column name to read text from (for --csv)")
    parser.add_argument("--out", default="milestone1_report", help="Output filename prefix (one-shot mode)")
    args = parser.parse_args()

    if args.text or args.txt or args.csv:
        run_one_shot(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()