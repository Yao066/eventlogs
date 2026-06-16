#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute AgentSimulator evaluation metrics using log_distance_measures.

This script follows the metric calculation style in evaluation.ipynb:

- n_gram_distribution_distance(..., n=2)
- n_gram_distribution_distance(..., n=3)
- absolute_event_distribution_distance(..., discretize_type=BOTH, discretize_event=discretize_to_hour)
- relative_event_distribution_distance(..., discretize_type=BOTH, discretize_event=discretize_to_hour)
- circadian_event_distribution_distance(..., discretize_type=BOTH)
- cycle_time_distribution_distance(..., bin_size=pd.Timedelta(hours=1))

Input:
    --ref: test / ground-truth CSV
    --sim: folder containing simulated_log_0.csv ... simulated_log_9.csv

Output:
    <sim folder>/agent_evaluation_metrics.csv
    <sim folder>/agent_evaluation_metrics_summary.csv


python compute_agent_metrics.py \
    --ref ../AgentSimulator/simulated_data/production/test_preprocessed.csv \
    --sim ../AgentSimulator/simulated_data/Production/ \
    --case-col case_id \
    --activity-col activity_name \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --resource-col resource \
    --sim-case-col case_id \
    --sim-activity-col activity_name \
    --sim-start-col start_timestamp \
    --sim-end-col end_timestamp \
    --sim-resource-col resource

BPI：
python compute_agent_metrics.py \
    --ref ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/test_preprocessed.csv \
    --sim ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/ \
    --case-col "case_id" \
    --activity-col "activity_name" \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --resource-col "resource" \
    --sim-case-col case_id \
    --sim-activity-col activity_name \
    --sim-start-col start_timestamp \
    --sim-end-col end_timestamp \
    --sim-resource-col resource


Production：
python compute_agent_metrics.py \
    --ref ../AgentSimulator/simulated_data/Production_Data_processed/autonomous/test_preprocessed.csv \
    --sim ../AgentSimulator/simulated_data/Production_Data_processed/autonomous/ \
    --case-col "case_id" \
    --activity-col "activity" \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --resource-col "resource" \
    --sim-case-col case_id \
    --sim-activity-col activity_name \
    --sim-start-col start_timestamp \
    --sim-end-col end_timestamp \
    --sim-resource-col resource

python compute_agent_metrics.py \
    --ref ../simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E/best_result/evaluation/test_log.csv \
    --sim ../simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E/best_result/evaluation \
    --case-col "case_id" \
    --activity-col "activity_name" \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --resource-col "resource" \
    --sim-case-col case_id \
    --sim-activity-col activity \
    --sim-start-col start_time \
    --sim-end-col end_time \
    --sim-resource-col resource
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from log_distance_measures.config import (
    EventLogIDs,
    AbsoluteTimestampType,
    discretize_to_hour,
)
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance


# ---------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------

CANONICAL_CASE = "case_id"
CANONICAL_ACTIVITY = "activity"
CANONICAL_START = "start_time"
CANONICAL_END = "end_time"
CANONICAL_RESOURCE = "resource"


def auto_align_common_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align common column names to:
        case_id, activity, resource, start_time, end_time

    This follows the style in the uploaded evaluation.ipynb.
    Explicit CLI column mappings are still applied afterwards.
    """
    df = df.copy()

    if "case:concept:name" in df.columns:
        df = df.rename(columns={"case:concept:name": CANONICAL_CASE})
    elif "caseid" in df.columns:
        df = df.rename(columns={"caseid": CANONICAL_CASE})
    elif "Case ID" in df.columns:
        df = df.rename(columns={"Case ID": CANONICAL_CASE})
    elif "case" in df.columns and CANONICAL_CASE not in df.columns:
        df = df.rename(columns={"case": CANONICAL_CASE})

    if "Activity" in df.columns:
        df = df.rename(columns={"Activity": CANONICAL_ACTIVITY})
    elif "activity_name" in df.columns:
        df = df.rename(columns={"activity_name": CANONICAL_ACTIVITY})
    elif "task" in df.columns:
        df = df.rename(columns={"task": CANONICAL_ACTIVITY})
    elif "concept:name" in df.columns:
        df = df.rename(columns={"concept:name": CANONICAL_ACTIVITY})

    if "Resource" in df.columns:
        df = df.rename(columns={"Resource": CANONICAL_RESOURCE})
    elif "user" in df.columns:
        df = df.rename(columns={"user": CANONICAL_RESOURCE})
    elif "agent" in df.columns:
        if CANONICAL_RESOURCE in df.columns:
            df = df.drop(columns=[CANONICAL_RESOURCE])
        df = df.rename(columns={"agent": CANONICAL_RESOURCE})
    elif "org:resource" in df.columns:
        df = df.rename(columns={"org:resource": CANONICAL_RESOURCE})

    if "start_timestamp" in df.columns:
        df = df.rename(columns={"start_timestamp": CANONICAL_START})
    elif "start:timestamp" in df.columns:
        df = df.rename(columns={"start:timestamp": CANONICAL_START})

    if "end_timestamp" in df.columns:
        df = df.rename(columns={"end_timestamp": CANONICAL_END})
    elif "time:timestamp" in df.columns:
        df = df.rename(columns={"time:timestamp": CANONICAL_END})
    elif "complete_timestamp" in df.columns:
        df = df.rename(columns={"complete_timestamp": CANONICAL_END})
    elif "Complete timestamp" in df.columns:
        df = df.rename(columns={"Complete timestamp": CANONICAL_END})

    return df


def rename_explicit_columns(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    start_col: str,
    end_col: Optional[str],
    resource_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply explicit user-provided column mappings.

    If user passes columns that already equal canonical names, this is harmless.
    """
    df = df.copy()

    rename_map = {}

    if case_col in df.columns and case_col != CANONICAL_CASE:
        rename_map[case_col] = CANONICAL_CASE

    if activity_col in df.columns and activity_col != CANONICAL_ACTIVITY:
        rename_map[activity_col] = CANONICAL_ACTIVITY

    if start_col in df.columns and start_col != CANONICAL_START:
        rename_map[start_col] = CANONICAL_START

    if end_col and end_col in df.columns and end_col != CANONICAL_END:
        rename_map[end_col] = CANONICAL_END

    if resource_col and resource_col in df.columns and resource_col != CANONICAL_RESOURCE:
        rename_map[resource_col] = CANONICAL_RESOURCE

    df = df.rename(columns=rename_map)
    return df


def parse_datetime_column(
    series: pd.Series,
    datetime_format: Optional[str] = None,
    dayfirst: bool = False,
    utc: bool = True,
) -> pd.Series:
    """
    Robust timestamp parsing.

    The uploaded notebook uses:
        pd.to_datetime(..., utc=True, format='mixed')

    This function follows that behavior by default.
    """
    s = series.astype("string").str.strip()

    if datetime_format:
        return pd.to_datetime(
            s,
            format=datetime_format,
            errors="coerce",
            dayfirst=dayfirst,
            utc=utc,
        )

    # pandas >= 2 supports format="mixed"
    try:
        return pd.to_datetime(
            s,
            format="mixed",
            errors="coerce",
            dayfirst=dayfirst,
            utc=utc,
        )
    except Exception:
        return pd.to_datetime(
            s,
            errors="coerce",
            dayfirst=dayfirst,
            utc=utc,
        )


def first_invalid_row_report(
    original_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    path: str,
) -> str:
    checks = {
        CANONICAL_CASE: parsed_df[CANONICAL_CASE].isna() if CANONICAL_CASE in parsed_df.columns else pd.Series(True, index=parsed_df.index),
        CANONICAL_ACTIVITY: parsed_df[CANONICAL_ACTIVITY].isna() if CANONICAL_ACTIVITY in parsed_df.columns else pd.Series(True, index=parsed_df.index),
        CANONICAL_START: parsed_df[CANONICAL_START].isna() if CANONICAL_START in parsed_df.columns else pd.Series(True, index=parsed_df.index),
        CANONICAL_END: parsed_df[CANONICAL_END].isna() if CANONICAL_END in parsed_df.columns else pd.Series(True, index=parsed_df.index),
    }

    invalid_any = checks[CANONICAL_CASE] | checks[CANONICAL_ACTIVITY] | checks[CANONICAL_START] | checks[CANONICAL_END]

    if not invalid_any.any():
        return ""

    idx = invalid_any[invalid_any].index[0]
    csv_row_number = int(idx) + 2

    reasons = []
    for col, mask in checks.items():
        if bool(mask.loc[idx]):
            raw_value = original_df.loc[idx, col] if col in original_df.columns else "<COLUMN NOT FOUND>"
            reasons.append(f"- {col} invalid/missing/unparseable, value={raw_value!r}")

    raw_row = original_df.loc[idx].to_dict()

    return (
        f"\nFirst invalid row detail:\n"
        f"  file: {path}\n"
        f"  pandas_index: {idx}\n"
        f"  csv_row_number: {csv_row_number}\n"
        f"  reasons:\n"
        + "\n".join(f"    {r}" for r in reasons)
        + f"\n  raw row:\n"
        f"    {raw_row}\n"
    )


def read_event_log(
    path: str,
    case_col: str,
    activity_col: str,
    start_col: str,
    end_col: Optional[str],
    resource_col: Optional[str],
    sep: str = ",",
    encoding: str = "utf-8",
    datetime_format: Optional[str] = None,
    dayfirst: bool = False,
    utc: bool = True,
    drop_invalid: bool = False,
    allow_negative_duration: bool = True,
) -> pd.DataFrame:
    """
    Read CSV and convert it to the canonical columns expected by EventLogIDs.

    Important:
    - By default allow_negative_duration=True, because the official metric functions
      do not require us to reject these rows before metric calculation.
    - Invalid timestamp rows are still rejected unless --drop-invalid is used.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = pd.read_csv(csv_path, sep=sep, encoding=encoding)
    df = raw.copy()

    # First handle common names, then explicit mappings.
    df = auto_align_common_column_names(df)
    df = rename_explicit_columns(
        df,
        case_col=case_col,
        activity_col=activity_col,
        start_col=start_col,
        end_col=end_col,
        resource_col=resource_col,
    )

    required = [CANONICAL_CASE, CANONICAL_ACTIVITY, CANONICAL_START, CANONICAL_END]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required canonical columns after mapping: {missing}\n"
            f"Available columns after mapping: {list(df.columns)}\n"
            f"Original columns: {list(raw.columns)}"
        )

    # Keep original canonical-like df for reporting
    report_df = df.copy()

    df[CANONICAL_START] = parse_datetime_column(
        df[CANONICAL_START],
        datetime_format=datetime_format,
        dayfirst=dayfirst,
        utc=utc,
    )
    df[CANONICAL_END] = parse_datetime_column(
        df[CANONICAL_END],
        datetime_format=datetime_format,
        dayfirst=dayfirst,
        utc=utc,
    )

    invalid_mask = (
        df[CANONICAL_CASE].isna()
        | df[CANONICAL_ACTIVITY].isna()
        | df[CANONICAL_START].isna()
        | df[CANONICAL_END].isna()
    )

    negative_duration_mask = df[CANONICAL_END] < df[CANONICAL_START]

    validation_bad_duration = negative_duration_mask
    if allow_negative_duration:
        validation_bad_duration = pd.Series(False, index=df.index)

    if (invalid_mask.any() or validation_bad_duration.any()) and not drop_invalid:
        field_counts = {
            "case_missing": int(df[CANONICAL_CASE].isna().sum()),
            "activity_missing": int(df[CANONICAL_ACTIVITY].isna().sum()),
            "start_time_missing_or_unparseable": int(df[CANONICAL_START].isna().sum()),
            "end_time_missing_or_unparseable": int(df[CANONICAL_END].isna().sum()),
            "end_time_before_start_time": int(negative_duration_mask.sum()),
        }

        raise ValueError(
            f"{path} has invalid rows.\n"
            f"Invalid required field rows: {int(invalid_mask.sum())}; "
            f"end_time < start_time rows: {int(negative_duration_mask.sum())}.\n"
            f"Field-level counts: {field_counts}\n"
            f"{first_invalid_row_report(report_df, df, path)}\n"
            f"Use --drop-invalid to drop rows with missing/unparseable fields. "
            f"Use --disallow-negative-duration if you want end_time < start_time to be treated as invalid."
        )

    if drop_invalid:
        valid_mask = ~invalid_mask & ~validation_bad_duration
        df = df.loc[valid_mask].copy()

    # Ensure expected types
    df[CANONICAL_CASE] = df[CANONICAL_CASE].astype(str)
    df[CANONICAL_ACTIVITY] = df[CANONICAL_ACTIVITY].astype(str)

    if CANONICAL_RESOURCE not in df.columns:
        df[CANONICAL_RESOURCE] = "UNKNOWN"
    else:
        df[CANONICAL_RESOURCE] = df[CANONICAL_RESOURCE].fillna("UNKNOWN").astype(str)

    # Sort like the notebook / process mining convention.
    df = df.sort_values(
        by=[CANONICAL_CASE, CANONICAL_START, CANONICAL_END],
        kind="mergesort",
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path} contains no valid events.")

    return df


# ---------------------------------------------------------------------
# Metric calculation using log_distance_measures
# ---------------------------------------------------------------------

def make_event_log_ids() -> EventLogIDs:
    return EventLogIDs(
        case=CANONICAL_CASE,
        activity=CANONICAL_ACTIVITY,
        start_time=CANONICAL_START,
        end_time=CANONICAL_END,
        resource=CANONICAL_RESOURCE,
    )


def compute_metrics_for_pair(
    test_log: pd.DataFrame,
    simulated_log: pd.DataFrame,
    event_log_ids: EventLogIDs,
) -> Dict[str, float]:
    """
    Metric calculation follows evaluation.ipynb, plus 2-gram for compatibility
    with Simod-like evaluation_metrics.csv.
    """
    two_gram = n_gram_distribution_distance(
        test_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        n=2,
    )

    three_gram = n_gram_distribution_distance(
        test_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        n=3,
    )

    absolute = absolute_event_distribution_distance(
        test_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour,
    )

    relative = relative_event_distribution_distance(
        test_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour,
    )

    circadian = circadian_event_distribution_distance(
        test_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        discretize_type=AbsoluteTimestampType.BOTH,
    )

    cycle_time = cycle_time_distribution_distance(
        test_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        bin_size=pd.Timedelta(hours=1),
    )

    return {
        "two_gram_distance": float(two_gram),
        "three_gram_distance": float(three_gram),
        "absolute_event_distribution": float(absolute),
        "relative_event_distribution": float(relative),
        "circadian_event_distribution": float(circadian),
        "cycle_time_distribution": float(cycle_time),
    }


def metrics_to_long_rows(run_num: int, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    order = [
        "three_gram_distance",
        "two_gram_distance",
        "absolute_event_distribution",
        "relative_event_distribution",
        "circadian_event_distribution",
        "cycle_time_distribution",
    ]

    rows = []
    for metric in order:
        rows.append(
            {
                "run_num": run_num,
                "metric": metric,
                "distance": metrics[metric],
            }
        )
    return rows


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df
        .groupby("metric", as_index=False)["distance"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute AgentSimulator evaluation metrics for simulated_log_0.csv "
            "to simulated_log_9.csv using log_distance_measures."
        )
    )

    parser.add_argument("--ref", required=True, help="Reference/test/ground-truth CSV file.")
    parser.add_argument(
        "--sim",
        required=True,
        help="Folder containing simulated_log_0.csv ... simulated_log_9.csv.",
    )

    parser.add_argument("--case-col", required=True, help="Case id column name for reference log.")
    parser.add_argument("--activity-col", required=True, help="Activity column name for reference log.")
    parser.add_argument("--start-col", required=True, help="Start timestamp column name for reference log.")
    parser.add_argument("--end-col", required=True, help="End timestamp column name for reference log.")
    parser.add_argument("--resource-col", default=None, help="Resource column name for reference log. Optional.")

    parser.add_argument("--sim-case-col", default=None, help="Case id column for simulated logs. Defaults to --case-col.")
    parser.add_argument("--sim-activity-col", default=None, help="Activity column for simulated logs. Defaults to --activity-col.")
    parser.add_argument("--sim-start-col", default=None, help="Start timestamp column for simulated logs. Defaults to --start-col.")
    parser.add_argument("--sim-end-col", default=None, help="End timestamp column for simulated logs. Defaults to --end-col.")
    parser.add_argument("--sim-resource-col", default=None, help="Resource column for simulated logs. Defaults to --resource-col.")

    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of simulated logs. Default: 10.",
    )
    parser.add_argument(
        "--sim-file-prefix",
        default="simulated_log_",
        help="Prefix of simulated log files. Default: simulated_log_",
    )
    parser.add_argument(
        "--sim-file-ext",
        default=".csv",
        help="Extension of simulated log files. Default: .csv",
    )

    parser.add_argument("--sep", default=",", help="CSV separator. Default: comma.")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding. Default: utf-8.")
    parser.add_argument(
        "--datetime-format",
        default=None,
        help="Optional datetime format. If omitted, uses pandas format='mixed'.",
    )
    parser.add_argument("--dayfirst", action="store_true", help="Parse dates with day first.")
    parser.add_argument(
        "--utc",
        action="store_true",
        help="Parse timestamps as UTC. Recommended. If omitted, script still defaults to UTC for compatibility.",
    )

    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop rows with missing/unparseable required fields.",
    )

    parser.add_argument(
        "--disallow-negative-duration",
        action="store_true",
        help=(
            "Treat end_time < start_time as invalid. "
            "By default, this script allows negative durations to match the notebook-style calculation more closely."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path for per-run metrics. "
            "Default: <sim folder>/agent_evaluation_metrics.csv"
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    sim_dir = Path(args.sim)
    if not sim_dir.exists():
        print(f"ERROR: simulated log folder does not exist: {sim_dir}", file=sys.stderr)
        return 1

    if not sim_dir.is_dir():
        print(f"ERROR: --sim should be a folder path, but got a file: {sim_dir}", file=sys.stderr)
        return 1

    sim_case_col = args.sim_case_col or args.case_col
    sim_activity_col = args.sim_activity_col or args.activity_col
    sim_start_col = args.sim_start_col or args.start_col
    sim_end_col = args.sim_end_col or args.end_col
    sim_resource_col = args.sim_resource_col or args.resource_col

    # The notebook uses utc=True. Keep UTC enabled even if user forgets --utc.
    use_utc = True

    allow_negative_duration = not args.disallow_negative_duration

    event_log_ids = make_event_log_ids()

    try:
        test_log = read_event_log(
            path=args.ref,
            case_col=args.case_col,
            activity_col=args.activity_col,
            start_col=args.start_col,
            end_col=args.end_col,
            resource_col=args.resource_col,
            sep=args.sep,
            encoding=args.encoding,
            datetime_format=args.datetime_format,
            dayfirst=args.dayfirst,
            utc=use_utc,
            drop_invalid=args.drop_invalid,
            allow_negative_duration=allow_negative_duration,
        )

        all_rows: List[Dict[str, Any]] = []

        for run_num in range(args.num_runs):
            sim_file = sim_dir / f"{args.sim_file_prefix}{run_num}{args.sim_file_ext}"

            if not sim_file.exists():
                raise FileNotFoundError(f"Expected simulated log file not found: {sim_file}")

            print(f"Processing run {run_num}: {sim_file}", file=sys.stderr)

            simulated_log = read_event_log(
                path=str(sim_file),
                case_col=sim_case_col,
                activity_col=sim_activity_col,
                start_col=sim_start_col,
                end_col=sim_end_col,
                resource_col=sim_resource_col,
                sep=args.sep,
                encoding=args.encoding,
                datetime_format=args.datetime_format,
                dayfirst=args.dayfirst,
                utc=use_utc,
                drop_invalid=args.drop_invalid,
                allow_negative_duration=allow_negative_duration,
            )

            metrics = compute_metrics_for_pair(
                test_log=test_log,
                simulated_log=simulated_log,
                event_log_ids=event_log_ids,
            )

            all_rows.extend(metrics_to_long_rows(run_num, metrics))

        metrics_df = pd.DataFrame(all_rows, columns=["run_num", "metric", "distance"])

        output_csv = Path(args.output) if args.output else sim_dir / "agent_evaluation_metrics.csv"
        metrics_df.to_csv(output_csv, index=False)

        summary_df = summarize_metrics(metrics_df)
        summary_csv = sim_dir / "agent_evaluation_metrics_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        print(f"\nSaved per-run metrics to: {output_csv}")
        print(f"Saved summary metrics to: {summary_csv}")

        print("\nSummary:")
        print(summary_df.to_string(index=False))

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())