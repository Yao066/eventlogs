#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute process simulation log distances:
- NGD: N-Gram Distance
- AED: Absolute Event Distribution distance
- CED: Circadian Event Distribution distance
- RED: Relative Event Distribution distance
- CTD: Cycle Time Distribution distance

Input:
    Two CSV event logs:
    1) reference / test / ground-truth log
    2) simulated log

Each row is one event. Required columns are configurable:
    case id, activity, start timestamp, end timestamp

Example:
    python compute_log_distances.py \
        --ref test_log.csv \
        --sim simod_simulated_log.csv \
        --case-col case_id \
        --activity-col activity \
        --start-col start_time \
        --end-col end_time

If the simulated log uses different column names:
    python compute_log_distances.py \
        --ref test_log.csv \
        --sim agent_sim_log.csv \
        --case-col CaseID \
        --activity-col Activity \
        --start-col StartTime \
        --end-col EndTime \
        --sim-case-col case_id \
        --sim-activity-col task \
        --sim-start-col start_timestamp \
        --sim-end-col end_timestamp

python compute_log_distances.py \
    --ref ../simod_workspace/resources/event_logs/BPIChallenge2019_3WayMatchingEC_processed.csv \
    --sim ../simod_workspace/outputs/20260608_181447_24D690D3_ECDF_42D3_B1BA_9CA0EB114459/best_result/evaluation/simulated_log_0.csv \
    --case-col "case:concept:name" \
    --activity-col "concept:name" \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --sim-case-col case_id \
    --sim-activity-col activity \
    --sim-start-col start_time \
    --sim-end-col end_time


python compute_log_distances.py \
    --ref ../AgentSimulator/raw_data/BPIChallenge2019_3WayMatchingEC_processed.csv \
    --sim ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/simulated_log_0.csv \
    --case-col "case:concept:name" \
    --activity-col "concept:name" \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --sim-case-col case_id \
    --sim-activity-col activity_name \
    --sim-start-col start_timestamp \
    --sim-end-col end_timestamp \
    --utc \
    --drop-invalid


python compute_log_distances.py \
    --ref ../AgentSimulator/raw_data/production.csv \
    --sim ../AgentSimulator/simulated_data/Production-simod/ \
    --case-col case_id \
    --activity-col activity_name \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --sim-case-col case_id \
    --sim-activity-col activity \
    --sim-start-col start_time \
    --sim-end-col end_time \
    --utc \
    --drop-invalid
"""



from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd


NANOSECONDS_PER_HOUR = 3_600_000_000_000


# ---------------------------------------------------------------------
# CSV reading and validation
# ---------------------------------------------------------------------

def parse_datetime_series(
    s: pd.Series,
    datetime_format: str | None = None,
    dayfirst: bool = False,
    utc: bool = False,
) -> pd.Series:
    # 先统一转成字符串，避免某些列混有 datetime/object/NaN
    s_str = s.astype("string").str.strip()

    # 如果用户显式传了格式，优先用用户格式
    if datetime_format:
        return pd.to_datetime(
            s_str,
            format=datetime_format,
            errors="coerce",
            dayfirst=dayfirst,
            utc=utc,
        )

    # 优先尝试 ISO8601，适合 2018-10-16 08:19:17.373824548+00:00
    try:
        return pd.to_datetime(
            s_str,
            format="ISO8601",
            errors="coerce",
            dayfirst=dayfirst,
            utc=True,   # 推荐强制 UTC，避免 timezone-aware/naive 混合
        )
    except Exception:
        pass

    # 再尝试 mixed，适合同一列有多种时间格式
    try:
        return pd.to_datetime(
            s_str,
            format="mixed",
            errors="coerce",
            dayfirst=dayfirst,
            utc=True,
        )
    except Exception:
        pass

    # 最后回退到普通自动解析
    return pd.to_datetime(
        s_str,
        errors="coerce",
        dayfirst=dayfirst,
        utc=True,
    )

def _format_value(v) -> str:
    if pd.isna(v):
        return "<MISSING/NaN>"
    return repr(v)


def _build_first_invalid_report(
    raw: pd.DataFrame,
    df: pd.DataFrame,
    path: str,
    case_col: str,
    activity_col: str,
    start_col: str,
    end_col: str | None,
) -> str:
    """
    Build a detailed report for the first invalid CSV data row.
    CSV row number assumes the first row is the header.
    """
    invalid_case = raw[case_col].isna()
    invalid_activity = raw[activity_col].isna()
    invalid_start = df["_start_time"].isna()

    if end_col:
        invalid_end = df["_end_time"].isna()
    else:
        invalid_end = pd.Series(False, index=raw.index)

    bad_duration = df["_end_time"] < df["_start_time"]

    invalid_any = (
        invalid_case
        | invalid_activity
        | invalid_start
        | invalid_end
        | bad_duration
    )

    invalid_indices = raw.index[invalid_any].tolist()
    if not invalid_indices:
        return ""

    idx = invalid_indices[0]
    csv_row_number = int(idx) + 2

    reasons = []

    if invalid_case.loc[idx]:
        reasons.append(
            f"- case field invalid: column={case_col!r}, value={_format_value(raw.loc[idx, case_col])}"
        )

    if invalid_activity.loc[idx]:
        reasons.append(
            f"- activity field invalid: column={activity_col!r}, value={_format_value(raw.loc[idx, activity_col])}"
        )

    if invalid_start.loc[idx]:
        reasons.append(
            f"- start timestamp invalid/unparseable: column={start_col!r}, "
            f"value={_format_value(raw.loc[idx, start_col])}"
        )

    if end_col and invalid_end.loc[idx]:
        reasons.append(
            f"- end timestamp invalid/unparseable: column={end_col!r}, "
            f"value={_format_value(raw.loc[idx, end_col])}"
        )

    if bad_duration.loc[idx]:
        reasons.append(
            f"- negative duration: end_time < start_time; "
            f"start={_format_value(raw.loc[idx, start_col])}, "
            f"end={_format_value(raw.loc[idx, end_col]) if end_col else '<end_col not provided>'}"
        )

    raw_row = raw.loc[idx].to_dict()

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
    end_col: str | None,
    sep: str = ",",
    encoding: str = "utf-8",
    datetime_format: str | None = None,
    dayfirst: bool = False,
    utc: bool = False,
    drop_invalid: bool = False,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = pd.read_csv(csv_path, sep=sep, encoding=encoding)

    required_cols = [case_col, activity_col, start_col]
    if end_col:
        required_cols.append(end_col)

    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    df = pd.DataFrame(index=raw.index)
    df["_case"] = raw[case_col]
    df["_activity"] = raw[activity_col]
    df["_start_time"] = parse_datetime_series(
        raw[start_col],
        datetime_format=datetime_format,
        dayfirst=dayfirst,
        utc=utc,
    )

    if end_col:
        df["_end_time"] = parse_datetime_series(
            raw[end_col],
            datetime_format=datetime_format,
            dayfirst=dayfirst,
            utc=utc,
        )
    else:
        df["_end_time"] = df["_start_time"]

    invalid_case = df["_case"].isna()
    invalid_activity = df["_activity"].isna()
    invalid_start = df["_start_time"].isna()
    invalid_end = df["_end_time"].isna()
    bad_duration_mask = df["_end_time"] < df["_start_time"]

    invalid_required_mask = (
        invalid_case
        | invalid_activity
        | invalid_start
        | invalid_end
    )

    total_invalid_required = int(invalid_required_mask.sum())
    total_bad_duration = int(bad_duration_mask.sum())

    if (total_invalid_required > 0 or total_bad_duration > 0) and not drop_invalid:
        detail = _build_first_invalid_report(
            raw=raw,
            df=df,
            path=path,
            case_col=case_col,
            activity_col=activity_col,
            start_col=start_col,
            end_col=end_col,
        )

        field_counts = {
            "case_missing": int(invalid_case.sum()),
            "activity_missing": int(invalid_activity.sum()),
            "start_time_missing_or_unparseable": int(invalid_start.sum()),
            "end_time_missing_or_unparseable": int(invalid_end.sum()),
            "end_time_before_start_time": int(bad_duration_mask.sum()),
        }

        raise ValueError(
            f"{path} has invalid rows.\n"
            f"Invalid required fields rows: {total_invalid_required}; "
            f"end_time < start_time rows: {total_bad_duration}.\n"
            f"Field-level counts: {field_counts}\n"
            f"{detail}"
            f"Use --drop-invalid to drop these rows, but only do this if the invalid count is small."
        )

    if drop_invalid:
        valid_mask = ~invalid_required_mask & ~bad_duration_mask
        raw = raw.loc[valid_mask].copy()
        df = df.loc[valid_mask].copy()

    df["_case"] = df["_case"].astype(str)
    df["_activity"] = df["_activity"].astype(str)

    df = df.sort_values(
        ["_case", "_start_time", "_end_time", "_activity"],
        kind="mergesort",
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path} contains no valid events after parsing.")

    return df

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def bin_values(values: np.ndarray, bin_hours: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if bin_hours <= 0:
        return values
    return np.floor(values / bin_hours) * bin_hours


def to_hours_since(series: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """
    Convert pandas datetime Series to hours since origin.
    """
    ns = series.astype("int64").to_numpy()
    origin_ns = pd.Timestamp(origin).value
    return (ns - origin_ns) / NANOSECONDS_PER_HOUR


def wasserstein_1d(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute 1D Wasserstein distance / Earth Mover's Distance
    between two empirical distributions with equal weight per sample.

    This implementation avoids scipy dependency.

    Returns distance in the same unit as x/y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 and len(y) == 0:
        return 0.0
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Cannot compute Wasserstein distance with one empty distribution.")

    ux, cx = np.unique(x, return_counts=True)
    uy, cy = np.unique(y, return_counts=True)

    grid = np.unique(np.concatenate([ux, uy]))
    if len(grid) <= 1:
        return 0.0

    px = np.zeros(len(grid), dtype=float)
    py = np.zeros(len(grid), dtype=float)

    px[np.searchsorted(grid, ux)] = cx / cx.sum()
    py[np.searchsorted(grid, uy)] = cy / cy.sum()

    cdf_diff = np.cumsum(px - py)
    distance = np.sum(np.abs(cdf_diff[:-1]) * np.diff(grid))
    return float(distance)


def timestamp_series(df: pd.DataFrame, mode: str) -> pd.Series:
    """
    mode:
        start: use start timestamps only
        end: use end timestamps only
        both: use both start and end timestamps
    """
    parts = []
    if mode in ("start", "both"):
        parts.append(df["_start_time"])
    if mode in ("end", "both"):
        parts.append(df["_end_time"])
    if not parts:
        raise ValueError(f"Unknown timestamp mode: {mode}")
    return pd.concat(parts, ignore_index=True).dropna()


def log_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "events": int(len(df)),
        "cases": int(df["_case"].nunique()),
        "activities": int(df["_activity"].nunique()),
        "start_time_min": str(df["_start_time"].min()),
        "end_time_max": str(df["_end_time"].max()),
    }


# ---------------------------------------------------------------------
# NGD: N-Gram Distance
# ---------------------------------------------------------------------

def get_activity_traces(df: pd.DataFrame) -> List[List[str]]:
    ordered = df.sort_values(
        ["_case", "_start_time", "_end_time", "_activity"],
        kind="mergesort",
    )
    return ordered.groupby("_case", sort=False)["_activity"].apply(list).tolist()


def ngrams_from_trace(
    trace: Sequence[str],
    n: int,
    include_boundaries: bool = True,
) -> Iterable[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive.")

    seq = list(trace)
    if include_boundaries:
        seq = ["<START>"] + seq + ["<END>"]

    if len(seq) < n:
        return

    for i in range(len(seq) - n + 1):
        yield tuple(seq[i:i + n])


def ngram_distribution(df: pd.DataFrame, n: int, include_boundaries: bool = True) -> Counter:
    counts = Counter()
    for trace in get_activity_traces(df):
        counts.update(ngrams_from_trace(trace, n, include_boundaries))
    return counts


def ngram_distance(
    ref_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    n: int = 2,
    include_boundaries: bool = True,
) -> float:
    """
    NGD as total variation distance between normalized n-gram distributions.

    Range:
        0 = same n-gram distribution
        1 = completely disjoint n-gram distribution
    """
    ref_counts = ngram_distribution(ref_df, n=n, include_boundaries=include_boundaries)
    sim_counts = ngram_distribution(sim_df, n=n, include_boundaries=include_boundaries)

    ref_total = sum(ref_counts.values())
    sim_total = sum(sim_counts.values())

    if ref_total == 0 or sim_total == 0:
        raise ValueError("Cannot compute NGD because one log has no n-grams.")

    keys = set(ref_counts) | set(sim_counts)
    l1 = 0.0
    for k in keys:
        p = ref_counts.get(k, 0) / ref_total
        q = sim_counts.get(k, 0) / sim_total
        l1 += abs(p - q)

    return float(0.5 * l1)


# ---------------------------------------------------------------------
# AED: Absolute Event Distribution
# ---------------------------------------------------------------------

def absolute_event_distribution_distance(
    ref_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    timestamp_mode: str = "both",
    bin_hours: float = 1.0,
) -> float:
    ref_ts = timestamp_series(ref_df, timestamp_mode)
    sim_ts = timestamp_series(sim_df, timestamp_mode)

    origin = min(ref_ts.min(), sim_ts.min())

    ref_values = bin_values(to_hours_since(ref_ts, origin), bin_hours)
    sim_values = bin_values(to_hours_since(sim_ts, origin), bin_hours)

    return wasserstein_1d(ref_values, sim_values)


# ---------------------------------------------------------------------
# CED: Circadian Event Distribution
# ---------------------------------------------------------------------

def weekday_hour_values(
    df: pd.DataFrame,
    timestamp_mode: str = "both",
    bin_hours: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    ts = timestamp_series(df, timestamp_mode)

    weekdays = ts.dt.weekday.to_numpy()  # Monday=0, Sunday=6

    hours = (
        ts.dt.hour.to_numpy(dtype=float)
        + ts.dt.minute.to_numpy(dtype=float) / 60.0
        + ts.dt.second.to_numpy(dtype=float) / 3600.0
        + ts.dt.microsecond.to_numpy(dtype=float) / 3_600_000_000.0
        + ts.dt.nanosecond.to_numpy(dtype=float) / 3_600_000_000_000.0
    )

    hours = bin_values(hours, bin_hours)
    return weekdays, hours


def circadian_event_distribution_distance(
    ref_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    timestamp_mode: str = "both",
    bin_hours: float = 1.0,
    empty_day_penalty_hours: float = 24.0,
) -> float:
    """
    For each weekday, compare hour-of-day distributions.
    Then average over 7 weekdays.

    If one log has events on a weekday and the other has none,
    we assign empty_day_penalty_hours.
    """
    ref_weekday, ref_hour = weekday_hour_values(ref_df, timestamp_mode, bin_hours)
    sim_weekday, sim_hour = weekday_hour_values(sim_df, timestamp_mode, bin_hours)

    distances = []
    for day in range(7):
        x = ref_hour[ref_weekday == day]
        y = sim_hour[sim_weekday == day]

        if len(x) == 0 and len(y) == 0:
            distances.append(0.0)
        elif len(x) == 0 or len(y) == 0:
            distances.append(float(empty_day_penalty_hours))
        else:
            distances.append(wasserstein_1d(x, y))

    return float(np.mean(distances))


# ---------------------------------------------------------------------
# RED: Relative Event Distribution
# ---------------------------------------------------------------------

def relative_event_values(
    df: pd.DataFrame,
    timestamp_mode: str = "both",
    bin_hours: float = 1.0,
) -> np.ndarray:
    """
    Convert each event timestamp to hours since the start of its case.
    """
    case_start = df.groupby("_case")["_start_time"].transform("min")

    parts = []
    if timestamp_mode in ("start", "both"):
        start_rel = (df["_start_time"] - case_start).dt.total_seconds() / 3600.0
        parts.append(start_rel)
    if timestamp_mode in ("end", "both"):
        end_rel = (df["_end_time"] - case_start).dt.total_seconds() / 3600.0
        parts.append(end_rel)

    values = pd.concat(parts, ignore_index=True).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    return bin_values(values, bin_hours)


def relative_event_distribution_distance(
    ref_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    timestamp_mode: str = "both",
    bin_hours: float = 1.0,
) -> float:
    ref_values = relative_event_values(ref_df, timestamp_mode, bin_hours)
    sim_values = relative_event_values(sim_df, timestamp_mode, bin_hours)
    return wasserstein_1d(ref_values, sim_values)


# ---------------------------------------------------------------------
# CTD: Cycle Time Distribution
# ---------------------------------------------------------------------

def cycle_time_values(df: pd.DataFrame, bin_hours: float = 1.0) -> np.ndarray:
    case_times = df.groupby("_case").agg(
        case_start=("_start_time", "min"),
        case_end=("_end_time", "max"),
    )
    values = (
        case_times["case_end"] - case_times["case_start"]
    ).dt.total_seconds().to_numpy(dtype=float) / 3600.0

    values = values[np.isfinite(values)]
    values = values[values >= 0]

    return bin_values(values, bin_hours)


def cycle_time_distribution_distance(
    ref_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    bin_hours: float = 1.0,
) -> float:
    ref_values = cycle_time_values(ref_df, bin_hours)
    sim_values = cycle_time_values(sim_df, bin_hours)
    return wasserstein_1d(ref_values, sim_values)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute NGD, AED, CED, RED, CTD between one reference event log "
            "and 10 simulated event logs in a folder."
        )
    )

    parser.add_argument("--ref", required=True, help="Reference/test/ground-truth CSV file.")

    parser.add_argument(
        "--sim",
        required=True,
        help=(
            "Path to folder containing simulated_log_0.csv ... simulated_log_9.csv. "
        ),
    )

    parser.add_argument("--case-col", required=True, help="Case id column name for reference log.")
    parser.add_argument("--activity-col", required=True, help="Activity column name for reference log.")
    parser.add_argument("--start-col", required=True, help="Start timestamp column name for reference log.")
    parser.add_argument("--end-col", default=None, help="End timestamp column name for reference log.")

    parser.add_argument("--sim-case-col", default=None, help="Case id column name for simulated logs. Defaults to --case-col.")
    parser.add_argument("--sim-activity-col", default=None, help="Activity column name for simulated logs. Defaults to --activity-col.")
    parser.add_argument("--sim-start-col", default=None, help="Start timestamp column name for simulated logs. Defaults to --start-col.")
    parser.add_argument("--sim-end-col", default=None, help="End timestamp column name for simulated logs. Defaults to --end-col.")

    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of simulated logs to read. Default: 10, i.e. simulated_log_0.csv to simulated_log_9.csv.",
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
        help=(
            "Optional datetime format, e.g. '%%Y-%%m-%%d %%H:%%M:%%S'. "
            "If omitted, pandas will infer it."
        ),
    )
    parser.add_argument("--dayfirst", action="store_true", help="Parse dates with day first, e.g. 31/12/2025.")
    parser.add_argument("--utc", action="store_true", help="Parse timestamps as UTC / convert timezone-aware values to UTC.")
    parser.add_argument("--drop-invalid", action="store_true", help="Drop rows with missing/unparseable times or negative durations.")

    parser.add_argument(
        "--timestamp-mode",
        choices=["start", "end", "both"],
        default="both",
        help="Which timestamps to use for AED/CED/RED. Default: both.",
    )

    parser.add_argument("--ngram-n", type=int, default=2, help="n for NGD n-grams. Default: 2.")

    parser.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Do not add <START>/<END> tokens when computing n-grams.",
    )

    parser.add_argument(
        "--bin-minutes",
        type=float,
        default=60.0,
        help="Time bin size in minutes for AED/CED/RED/CTD. Default: 60. Use 0 to disable binning.",
    )

    parser.add_argument(
        "--empty-day-penalty-hours",
        type=float,
        default=24.0,
        help="CED penalty if one log has events on a weekday and the other has none. Default: 24.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results.",
    )

    return parser

def compute_metrics_for_pair(
    ref_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    args: argparse.Namespace,
    bin_hours: float,
) -> Dict[str, float]:
    return {
        "NGD_2": ngram_distance(
            ref_df,
            sim_df,
            n=2,
            include_boundaries=not args.no_boundaries,
        ),
        "NGD_3": ngram_distance(
            ref_df,
            sim_df,
            n=3,
            include_boundaries=not args.no_boundaries,
        ),
        "AED_hours": absolute_event_distribution_distance(
            ref_df,
            sim_df,
            timestamp_mode=args.timestamp_mode,
            bin_hours=bin_hours,
        ),
        "RED_hours": relative_event_distribution_distance(
            ref_df,
            sim_df,
            timestamp_mode=args.timestamp_mode,
            bin_hours=bin_hours,
        ),
        "CED_hours": circadian_event_distribution_distance(
            ref_df,
            sim_df,
            timestamp_mode=args.timestamp_mode,
            bin_hours=bin_hours,
            empty_day_penalty_hours=args.empty_day_penalty_hours,
        ),
        "CTD_hours": cycle_time_distribution_distance(
            ref_df,
            sim_df,
            bin_hours=bin_hours,
        ),
    }

def metrics_to_csv_rows(run_num: int, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Convert internal metric names to Simod-like CSV metric names.
    """
    name_mapping = {
        "NGD_2": "two_gram_distance",
        "NGD_3": "three_gram_distance",
        "AED_hours": "absolute_event_distribution",
        "CED_hours": "circadian_event_distribution",
        "RED_hours": "relative_event_distribution",
        "CTD_hours": "cycle_time_distribution",
    }

    rows = []
    for internal_name, csv_name in name_mapping.items():
        if internal_name in metrics:
            rows.append(
                {
                    "run_num": run_num,
                    "metric": csv_name,
                    "distance": metrics[internal_name],
                }
            )

    return rows

def aggregate_metrics(per_run_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Return mean/std/min/max for each metric across runs.
    """
    if not per_run_metrics:
        raise ValueError("No per-run metrics to aggregate.")

    metric_names = per_run_metrics[0].keys()
    aggregated = {}

    for metric in metric_names:
        values = np.array([run[metric] for run in per_run_metrics], dtype=float)

        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return aggregated


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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
    sim_end_col = args.sim_end_col if args.sim_end_col is not None else args.end_col

    bin_hours = 0.0 if args.bin_minutes <= 0 else args.bin_minutes / 60.0

    try:
        ref_df = read_event_log(
            path=args.ref,
            case_col=args.case_col,
            activity_col=args.activity_col,
            start_col=args.start_col,
            end_col=args.end_col,
            sep=args.sep,
            encoding=args.encoding,
            datetime_format=args.datetime_format,
            dayfirst=args.dayfirst,
            utc=args.utc,
            drop_invalid=args.drop_invalid,
        )

        csv_rows = []
        per_run_metrics = []

        for i in range(args.num_runs):
            sim_file = sim_dir / f"{args.sim_file_prefix}{i}{args.sim_file_ext}"

            if not sim_file.exists():
                raise FileNotFoundError(
                    f"Expected simulated log file not found: {sim_file}"
                )

            print(f"Processing run {i}: {sim_file}", file=sys.stderr)

            sim_df = read_event_log(
                path=str(sim_file),
                case_col=sim_case_col,
                activity_col=sim_activity_col,
                start_col=sim_start_col,
                end_col=sim_end_col,
                sep=args.sep,
                encoding=args.encoding,
                datetime_format=args.datetime_format,
                dayfirst=args.dayfirst,
                utc=args.utc,
                drop_invalid=args.drop_invalid,
            )

            metrics = compute_metrics_for_pair(
                ref_df=ref_df,
                sim_df=sim_df,
                args=args,
                bin_hours=bin_hours,
            )

            per_run_metrics.append(metrics)
            csv_rows.extend(metrics_to_csv_rows(run_num=i, metrics=metrics))

        metrics_df = pd.DataFrame(csv_rows, columns=["run_num", "metric", "distance"])

        output_csv = sim_dir / "agent_evaluation_metrics.csv"
        metrics_df.to_csv(output_csv, index=False)

        aggregated = aggregate_metrics(per_run_metrics)

        mean_std_rows = []
        csv_metric_name_mapping = {
            "NGD_2": "two_gram_distance",
            "NGD_3": "three_gram_distance",
            "AED_hours": "absolute_event_distribution",
            "RED_hours": "relative_event_distribution",
            "CED_hours": "circadian_event_distribution",
            "CTD_hours": "cycle_time_distribution",
        }

        for internal_metric, stats in aggregated.items():
            mean_std_rows.append(
                {
                    "metric": csv_metric_name_mapping.get(internal_metric, internal_metric),
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )

        mean_std_df = pd.DataFrame(
            mean_std_rows,
            columns=["metric", "mean", "std", "min", "max"],
        )

        output_summary_csv = sim_dir / "agent_evaluation_metrics_summary.csv"
        mean_std_df.to_csv(output_summary_csv, index=False)

        print(f"\nSaved per-run metrics to: {output_csv}")
        print(f"Saved summary metrics to: {output_summary_csv}")

        print("\nSummary:")
        print(mean_std_df.to_string(index=False))

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    

if __name__ == "__main__":
    raise SystemExit(main())