import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BINS = 40

def find_wt_metric(df, metric_col, wt_id=None, wt_flag_col="is_wt"):
    if wt_id is not None and "variant_id" in df.columns:
        wt_row = df.loc[df["variant_id"].astype(str) == str(wt_id)]
        if not wt_row.empty:
            return float(wt_row.iloc[0][metric_col])

    if wt_flag_col is not None and wt_flag_col in df.columns:
        wt_row = df.loc[df[wt_flag_col] == 1]
        if len(wt_row) == 1:
            return float(wt_row.iloc[0][metric_col])

    return None

def plot_hist(values, metric_col, title, wt_metric=None, outfile=None, invert_x=True):
    values = values.astype(float)

    min_val = float(values.min())
    max_val = float(values.max())
    span = max_val - min_val
    pad = span * 0.05 if span > 0 else 0.01

    plt.figure(figsize=(10, 6))

    sns.histplot(
        values,
        kde=True,
        bins=BINS,
        edgecolor="black",
        alpha=0.85,
    )

    plt.xlabel(metric_col)
    plt.ylabel("Count")
    plt.title(title)

    plt.xlim(min_val - pad, max_val + pad)
    if invert_x:
        plt.gca().invert_xaxis()
    plt.ylim(bottom=0)

    if wt_metric is not None:
        plt.axvline(wt_metric, color="red", linestyle="--", linewidth=1.5, label="WT metric")
        plt.legend()

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=300)

    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to esm2_scores.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs")
    ap.add_argument("--metric", default="avg_log_prob", help="Metric column to plot")
    ap.add_argument("--step-col", default="n_mut_positions", help="Step column")
    ap.add_argument("--wt-id", default=None, help="WT variant_id (optional)")
    ap.add_argument("--wt-flag-col", default="is_wt", help="WT flag column (default: is_wt)")
    ap.add_argument("--no-invert-x", action="store_true", help="Do not invert x-axis")
    args = ap.parse_args()

    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    metric_col = args.metric
    step_col = args.step_col
    invert_x = not args.no_invert_x

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for col in [metric_col, step_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV!")

    wt_metric = find_wt_metric(df, metric_col, wt_id=args.wt_id, wt_flag_col=args.wt_flag_col)

    vals_all = df[metric_col].astype(float)
    plot_hist(
        vals_all,
        metric_col=metric_col,
        title=f"Distribution of {metric_col}",
        wt_metric=wt_metric,
        outfile=outdir / f"{metric_col}_ALL.png",
        invert_x=invert_x,
    )

    steps = sorted(df[step_col].dropna().unique())
    for step in steps:
        sub = df[df[step_col] == step]
        if sub.empty:
            continue
        values = sub[metric_col].astype(float)
        plot_hist(
            values,
            metric_col=metric_col,
            title=f"Distribution of {metric_col} — step={int(step)}",
            wt_metric=wt_metric,
            outfile=outdir / f"{metric_col}_step{int(step)}.png",
            invert_x=invert_x,
        )

if __name__ == "__main__":
    main()
