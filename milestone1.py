"""
Mood Mentor - Milestone 1 Interactive Session
=================================================
Task 1's flow explicitly calls for:
  Create/enter text input -> Upload .txt file -> Upload .csv file ->
  Read input data -> Validate input format -> Pass valid text to
  preprocessing -> Handle empty or invalid inputs -> Verify all
  supported input methods work correctly.

That's an interactive workflow, not a one-shot CLI argument. This
script runs a live menu loop: you feed it text/files as you go,
invalid input is caught and you're re-prompted (never crashes the
session), and every valid entry is accumulated into one running
session. At the end you generate ONE Milestone 1 report covering
everything you entered, saved to disk.

Usage:
    python3 run_milestone1.py
"""

import os
import sys

from text_ingestion import (
    ingest_text_input, ingest_txt_file, ingest_csv_file, IngestionError,
)
from pipeline import run_pipeline_on_record
from report_generator import build_report, save_report_csv, save_report_json


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

    csv_path = f"{out_name}.csv"
    json_path = f"{out_name}.json"
    save_report_csv(report, csv_path)
    save_report_json(report, json_path)

    s = report["summary"]
    print("\n" + "=" * 55)
    print(" MILESTONE 1 REPORT GENERATED")
    print("=" * 55)
    print(f" Samples analyzed : {s['num_samples_analyzed']}")
    print(f" Positive         : {s['positive_count']} ({s['positive_pct']}%)")
    print(f" Negative         : {s['negative_count']} ({s['negative_pct']}%)")
    print(f" Neutral          : {s['neutral_count']} ({s['neutral_pct']}%)")
    print("-" * 55)
    print(f" Saved to: {os.path.abspath(csv_path)}")
    print(f"           {os.path.abspath(json_path)}")
    print("=" * 55)
    print(" Open the .csv in Excel, or download it from your file")
    print(" explorer at the path above — that's your Milestone 1")
    print(" deliverable.")


SESSION_ROWS = []


def main():
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


if __name__ == "__main__":
    main()