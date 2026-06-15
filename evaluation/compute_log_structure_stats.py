#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute trace variants and case length statistics.

For test log:
    compute once.

For simulated logs:
    read simulated_log_0.csv ... simulated_log_9.csv from a folder,
    compute statistics per log, then output mean values.

Usage:
    python compute_log_structure_stats.py \
      --test ../AgentSimulator/raw_data/BPIChallenge2019_3WayMatchingEC_processed.csv \
      --sim-dir ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/ \
      --case-col "case:concept:name" \
      --activity-col "concept:name" \
      --start-col start_timestamp

python compute_log_structure_stats.py \
  --original-log ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/test_preprocessed.csv \
  --simod-dir ../simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E/best_result/evaluation \
  --agentsim-dir ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated \
  --original-case-col "case_id" \
  --original-activity-col "activity_name" \
  --original-start-col start_timestamp \
  --simod-case-col case_id \
  --simod-activity-col activity \
  --simod-start-col start_time \
  --agentsim-case-col case_id \
  --agentsim-activity-col activity_name \
  --agentsim-start-col start_timestamp


"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


STAT_ORDER = [
    "trace_variants",
    "mean_case_length",
    "median_case_length",
    "min_case_length",
    "max_case_length",
]


def read_log(
    path: Path,
    case_col: str,
    activity_col: str,
    start_col: str | None,
    sep: str,
    encoding: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, encoding=encoding)

    required = [case_col, activity_col]
    if start_col:
        required.append(start_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[case_col] = df[case_col].astype(str)
    df[activity_col] = df[activity_col].astype(str)

    if start_col:
        df[start_col] = pd.to_datetime(
            df[start_col],
            utc=True,
            format="mixed",
            errors="coerce",
        )

        invalid_rows = df[start_col].isna().sum()
        if invalid_rows > 0:
            print(
                f"WARNING: {path} has {invalid_rows} rows with invalid start timestamp. "
                f"These rows will be dropped.",
                file=sys.stderr,
            )
            df = df.dropna(subset=[start_col]).copy()

        df = df.sort_values([case_col, start_col], kind="mergesort")
    else:
        df = df.sort_values([case_col], kind="mergesort")

    return df


def compute_stats(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
) -> dict[str, float]:
    traces = df.groupby(case_col, sort=False)[activity_col].apply(tuple)
    case_lengths = traces.apply(len)

    return {
        "trace_variants": float(traces.nunique()),
        "mean_case_length": float(case_lengths.mean()),
        "median_case_length": float(case_lengths.median()),
        "min_case_length": float(case_lengths.min()),
        "max_case_length": float(case_lengths.max()),
    }


def compute_single_log_stats(
    log_path: Path,
    case_col: str,
    activity_col: str,
    start_col: str | None,
    sep: str,
    encoding: str,
) -> dict[str, float]:
    df = read_log(
        log_path,
        case_col=case_col,
        activity_col=activity_col,
        start_col=start_col,
        sep=sep,
        encoding=encoding,
    )

    return compute_stats(
        df,
        case_col=case_col,
        activity_col=activity_col,
    )


def compute_simulated_mean_stats(
    sim_dir: Path,
    num_runs: int,
    sim_file_prefix: str,
    sim_file_ext: str,
    case_col: str,
    activity_col: str,
    start_col: str | None,
    sep: str,
    encoding: str,
) -> dict[str, float]:
    all_stats = []

    for i in range(num_runs):
        sim_path = sim_dir / f"{sim_file_prefix}{i}{sim_file_ext}"

        if not sim_path.exists():
            raise FileNotFoundError(f"Simulated log not found: {sim_path}")

        stats = compute_single_log_stats(
            log_path=sim_path,
            case_col=case_col,
            activity_col=activity_col,
            start_col=start_col,
            sep=sep,
            encoding=encoding,
        )

        all_stats.append(stats)

    stats_df = pd.DataFrame(all_stats)

    return {
        stat: float(stats_df[stat].mean())
        for stat in STAT_ORDER
    }


def normalize_start_col(value: str) -> str | None:
    value = value.strip()
    return value if value else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare diversity and case length statistics for original log, Simod, and AgentSimulator."
    )

    parser.add_argument(
        "--original-log",
        required=True,
        help="Original / test event log CSV file.",
    )

    parser.add_argument(
        "--simod-dir",
        required=True,
        help="Directory containing Simod simulated logs.",
    )

    parser.add_argument(
        "--agentsim-dir",
        required=True,
        help="Directory containing AgentSimulator simulated logs.",
    )

    # Original / test log columns
    parser.add_argument("--original-case-col", default="case_id")
    parser.add_argument("--original-activity-col", default="activity")
    parser.add_argument("--original-start-col", default="start_timestamp")

    # Simod log columns
    parser.add_argument("--simod-case-col", default="case_id")
    parser.add_argument("--simod-activity-col", default="activity")
    parser.add_argument("--simod-start-col", default="start_timestamp")

    # AgentSimulator log columns
    parser.add_argument("--agentsim-case-col", default="case_id")
    parser.add_argument("--agentsim-activity-col", default="activity")
    parser.add_argument("--agentsim-start-col", default="start_timestamp")

    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of simulated logs in each directory. Default: 10.",
    )

    parser.add_argument(
        "--sim-file-prefix",
        default="simulated_log_",
        help="Simulated log filename prefix. Default: simulated_log_",
    )

    parser.add_argument(
        "--sim-file-ext",
        default=".csv",
        help="Simulated log filename extension. Default: .csv",
    )

    parser.add_argument(
        "--sep",
        default=",",
        help="CSV separator for all logs. Default: comma.",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding for all logs. Default: utf-8.",
    )

    parser.add_argument(
        "--output-filename",
        default="diversity_case_structure_comparison.csv",
        help="Output CSV filename. Saved into both Simod and AgentSimulator directories.",
    )

    args = parser.parse_args()

    original_log = Path(args.original_log)
    simod_dir = Path(args.simod_dir)
    agentsim_dir = Path(args.agentsim_dir)

    if not original_log.exists() or not original_log.is_file():
        print(f"ERROR: original log not found: {original_log}", file=sys.stderr)
        return 1

    if not simod_dir.exists() or not simod_dir.is_dir():
        print(f"ERROR: Simod directory not found: {simod_dir}", file=sys.stderr)
        return 1

    if not agentsim_dir.exists() or not agentsim_dir.is_dir():
        print(f"ERROR: AgentSimulator directory not found: {agentsim_dir}", file=sys.stderr)
        return 1

    try:
        original_start_col = normalize_start_col(args.original_start_col)
        simod_start_col = normalize_start_col(args.simod_start_col)
        agentsim_start_col = normalize_start_col(args.agentsim_start_col)

        print("Computing original event log statistics...")
        original_stats = compute_single_log_stats(
            log_path=original_log,
            case_col=args.original_case_col,
            activity_col=args.original_activity_col,
            start_col=original_start_col,
            sep=args.sep,
            encoding=args.encoding,
        )

        print("Computing Simod simulated log statistics...")
        simod_stats = compute_simulated_mean_stats(
            sim_dir=simod_dir,
            num_runs=args.num_runs,
            sim_file_prefix=args.sim_file_prefix,
            sim_file_ext=args.sim_file_ext,
            case_col=args.simod_case_col,
            activity_col=args.simod_activity_col,
            start_col=simod_start_col,
            sep=args.sep,
            encoding=args.encoding,
        )

        print("Computing AgentSimulator simulated log statistics...")
        agentsim_stats = compute_simulated_mean_stats(
            sim_dir=agentsim_dir,
            num_runs=args.num_runs,
            sim_file_prefix=args.sim_file_prefix,
            sim_file_ext=args.sim_file_ext,
            case_col=args.agentsim_case_col,
            activity_col=args.agentsim_activity_col,
            start_col=agentsim_start_col,
            sep=args.sep,
            encoding=args.encoding,
        )

        rows = []
        for stat in STAT_ORDER:
            rows.append(
                {
                    "statistic": stat,
                    "original_event_log": original_stats[stat],
                    "simod": simod_stats[stat],
                    "agentsimulator": agentsim_stats[stat],
                }
            )

        result_df = pd.DataFrame(rows)

        simod_output = simod_dir / args.output_filename
        agentsim_output = agentsim_dir / args.output_filename

        result_df.to_csv(simod_output, index=False)
        result_df.to_csv(agentsim_output, index=False)

        print()
        print(f"Saved to Simod directory:          {simod_output}")
        print(f"Saved to AgentSimulator directory: {agentsim_output}")
        print()
        print(result_df.to_string(index=False))

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())