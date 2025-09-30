"""Example script: generate sample CSV, run QA checks, and write a readable Excel report.

This script intentionally keeps the filename with a space (`python phase1_qa_tool.py`) because
the repository examples reference it. Run with:

    python "python phase1_qa_tool.py"

It reuses `qa_checks.py` to keep logic consistent with the Streamlit app.
"""

from pathlib import Path
import pandas as pd
import argparse
import tempfile
from datetime import datetime
import logging
import sys
from qa_checks import tg142_check, tg224_check, electron_beam_check, tps_check


def create_sample_csv(path: Path):
    data = {
        'output': [100, 98, 102],
        'flatness': [100, 102, 97],
        'symmetry': [100, 101, 98],
        'dose_uniformity': [100, 101, 98],
        'beam_energy': [6, 6.2, 5.8],
        'electron_output': [100, 102, 97],
        'PDD': [100, 99, 101],
        'planned_vs_measured': [100, 97, 102],
        'gamma_index': [98, 96, 94]
    }
    df = pd.DataFrame(data)
    try:
        df.to_csv(path, index=False)
        logging.info("Sample CSV created: %s", path)
        return path
    except PermissionError:
        # Fallback to temp directory
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        tmp = Path(tempfile.gettempdir()) / f"sample_data_{ts}.csv"
        df.to_csv(tmp, index=False)
        logging.warning("Permission denied writing to %s. Wrote sample CSV instead to: %s", path, tmp)
        return tmp


def run_all_checks(df: pd.DataFrame):
    results = []
    results += tg142_check(df)
    results += tg224_check(df)
    results += electron_beam_check(df)
    results += tps_check(df)
    return results


def write_readable_report(results, out_path: Path):
    df_results = pd.DataFrame(results)
    # Ensure the expected pivot key exists ('Sample' or 'index')
    if 'Sample' in df_results.columns:
        index_key = 'Sample'
    elif 'index' in df_results.columns:
        index_key = 'index'
        df_results = df_results.rename(columns={'index': 'Sample'})
    else:
        raise RuntimeError('results do not contain a sample/index column')

    # Pivot and write Excel safely
    df_report = df_results.pivot(index='Sample', columns='parameter', values='status')
    df_report.reset_index(inplace=True)

    # Write to Excel using openpyxl engine
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_report.to_excel(writer, index=False)
        logging.info("Readable QA report generated: %s", out_path)
        return out_path
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        tmp = Path(tempfile.gettempdir()) / f"qa_report_readable_{ts}.xlsx"
        with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
            df_report.to_excel(writer, index=False)
        logging.warning("Permission denied writing to %s. Wrote report instead to: %s", out_path, tmp)
        return tmp


def summary(results):
    df = pd.DataFrame(results)
    if df.empty:
        print("No results to summarize.")
        return
    summary = df.groupby(['parameter', 'status']).size().unstack(fill_value=0)
    print("\nSummary (counts by parameter and status):")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description='Run example Phase 1 QA checks')
    parser.add_argument('--csv', default='sample_data.csv', help='Path to sample CSV to create/use')
    parser.add_argument('--report', default='qa_report_readable.xlsx', help='Output Excel report path or filename')
    parser.add_argument('--out-dir', default='.', help='Directory where CSV and report will be written')
    parser.add_argument('--no-create', dest='create', action='store_false', help='Do not create sample CSV (use existing)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    report_path = out_dir / args.report

    # Ensure out_dir exists (if different from cwd)
    # Configure logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logging.warning("Permission denied creating out-dir: %s. Will attempt to write to temp directory instead.", out_dir)

    # If creating, write sample CSV into out_dir to keep outputs together
    if args.create:
        target_csv = out_dir / Path(args.csv).name
        created = create_sample_csv(target_csv)
        if created is not None:
            csv_path = Path(created)
        else:
            csv_path = target_csv

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path} (use --no-create to skip creation)")

    df = pd.read_csv(csv_path)
    results = run_all_checks(df)
    written = write_readable_report(results, report_path)
    # If write_readable_report returned a fallback path, show that to user
    if written is not None and Path(written) != report_path:
        print(f"Report written to fallback location: {written}")
    summary(results)


if __name__ == '__main__':
    main()

