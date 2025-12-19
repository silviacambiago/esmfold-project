"""
Plotting script for ESM2 scores (HPC Cluster Friendly).
Functionally identical to the local version, but adds CLI args and headless backend.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def find_wt_metric(df, metric_col, wt_flag_col="is_wt"):
    if wt_flag_col in df.columns:
        wt_row = df.loc[df[wt_flag_col] == 1]
        if len(wt_row) == 1:
            return float(wt_row.iloc[0][metric_col])
    return None

def plot_step(values, step, metric_col, wt_metric=None, outfile=None):
    values = values.astype(float)

    min_val = float(values.min())
    max_val = float(values.max())
    span = max_val - min_val
    pad = span * 0.05 if span > 0 else 0.01

    plt.figure(figsize=(10, 6))

    sns.histplot(
        values,
        kde=True,
        bins=40,
        edgecolor="black",
        alpha=0.85,
    )

    plt.xlabel(metric_col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {metric_col} — step={step}")

    plt.xlim(min_val - pad, max_val + pad)
    plt.ylim(bottom=0)
    plt.gca().invert_xaxis()

    if wt_metric is not None:
        plt.axvline(wt_metric, color="red", linestyle="--", linewidth=1.5, label="WT metric")
        plt.legend()

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")

    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to csv")
    parser.add_argument("--outdir", type=str, required=True, help="Output folder")
    parser.add_argument("--metric", type=str, default="avg_log_prob", help="Metric column")
    args = parser.parse_args()

    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    metric_col = args.metric
    step_col = "n_mut_positions"

    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Basic Checks
    if metric_col not in df.columns:
        sys.exit(f"Column '{metric_col}' not found.")

    # Handle missing step column gracefully
    if step_col not in df.columns:
        print(f"'{step_col}' not found. Treating all as step 0.")
        df[step_col] = 0

    wt_metric = find_wt_metric(df, metric_col)
    if wt_metric:
        print(f"WT Metric: {wt_metric:.4f}")

    steps = sorted(df[step_col].dropna().unique())

    for step in steps:
        sub_df = df[df[step_col] == step]
        if sub_df.empty: continue

        vals = sub_df[metric_col]
        out_name = outdir / f"{metric_col}_step{int(step)}.png"

        plot_step(vals, int(step), metric_col, wt_metric, out_name)

if __name__ == "__main__":
    main()