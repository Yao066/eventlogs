#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read evaluation_metrics.csv and compute mean distance for each metric.

Input CSV format:
    run_num,metric,distance

Usage:
    python average_evaluation_metrics.py --dir agent_output


python evaluation_simod_metrics.py \
  --dir ../simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E/best_result/evaluation \
  --input-filename evaluation_metrics.csv \
  --output-filename mean_metrics.csv

python evaluation_simod_metrics.py \
  --dir ../simod_workspace/outputs/20260614_085801_32C2CFAC_7561_403B_866F_180DC2AB529C/best_result/evaluation \
  --input-filename evaluation_metrics.csv \
  --output-filename mean_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute mean distance for each metric in evaluation_metrics.csv."
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing evaluation_metrics.csv.",
    )

    parser.add_argument(
        "--input-filename",
        default="evaluation_metrics.csv",
        help="Input CSV filename. Default: evaluation_metrics.csv.",
    )

    parser.add_argument(
        "--output-filename",
        default="evaluation_metrics_mean.csv",
        help="Output CSV filename. Default: evaluation_metrics_mean.csv.",
    )

    args = parser.parse_args()

    input_dir = Path(args.dir)
    input_csv = input_dir / args.input_filename
    output_csv = input_dir / args.output_filename

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 1

    if not input_csv.exists():
        print(f"ERROR: input file not found: {input_csv}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(input_csv)

        required_cols = {"metric", "distance"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            print(
                f"ERROR: input CSV is missing columns: {sorted(missing_cols)}\n"
                f"Available columns: {list(df.columns)}",
                file=sys.stderr,
            )
            return 1

        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df = df.dropna(subset=["distance"])

        if df.empty:
            print("ERROR: no valid distance values.", file=sys.stderr)
            return 1

        result = (
            df.groupby("metric", as_index=False)["distance"]
            .mean()
            .rename(columns={"distance": "mean_distance"})
        )

        metric_order = [
            "three_gram_distance",
            "two_gram_distance",
            "absolute_event_distribution",
            "relative_event_distribution",
            "circadian_event_distribution",
            "arrival_event_distribution",
            "cycle_time_distribution",
        ]

        result["metric_order"] = result["metric"].apply(
            lambda x: metric_order.index(x) if x in metric_order else len(metric_order)
        )

        result = (
            result.sort_values(["metric_order", "metric"])
            .drop(columns=["metric_order"])
            .reset_index(drop=True)
        )

        result.to_csv(output_csv, index=False)

        print(f"Saved mean metrics to: {output_csv}")
        print()
        print(result.to_string(index=False))

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())