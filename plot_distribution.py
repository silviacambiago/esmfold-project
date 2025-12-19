"""
This script:
  - Loads esm2_scores.csv produced by esm2_analysis.py
  - Checks required columns
  - Finds WT metric (via is_wt==1, or via WT_ID if provided)
  - For each mutation step k (n_mut_positions == k):
      * plots histogram (+ KDE if seaborn exists)
      * marks WT with a vertical red dashed line
      * y-axis starts at 0
      * x-axis is tight around the data
      * x-axis is inverted (values closer to 0 appear on the RIGHT)
  - Saves one PNG per step into OUTDIR
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = r"tests\esm2_ubiquitin_multi\esm2_scores.csv"
METRIC_COLUMN = "avg_log_prob"
STEP_COLUMN = "n_mut_positions"
WT_ID = None
WT_IS_FLAG = "is_wt"
OUTDIR = Path("plots_stepA_ubiquitin")
BINS = 40

def find_wt_metric(df):
    """Return WT metric value if we can identify WT, else None."""
    if WT_ID is not None and "variant_id" in df.columns:
        wt_row = df.loc[df["variant_id"].astype(str) == str(WT_ID)]
        if not wt_row.empty:
            return float(wt_row.iloc[0][METRIC_COLUMN])

    if WT_IS_FLAG is not None and WT_IS_FLAG in df.columns:
        wt_row = df.loc[df[WT_IS_FLAG] == 1]
        if len(wt_row) == 1:
            return float(wt_row.iloc[0][METRIC_COLUMN])

    return None

def plot_step(values, step, wt_metric=None, outfile=None):
    """Plot histogram for one step."""
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


    plt.xlabel(METRIC_COLUMN)
    plt.ylabel("Count")
    plt.title(f"Distribution of {METRIC_COLUMN} — step={step}")

    plt.xlim(min_val - pad, max_val + pad)

    plt.gca().invert_xaxis()

    plt.ylim(bottom=0)

    if wt_metric is not None:
        plt.axvline(wt_metric, color="red", linestyle="--", linewidth=1.5, label="WT metric")
        plt.legend()

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=300)

    plt.show()

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in [METRIC_COLUMN, STEP_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV!")

    vals_all = df[METRIC_COLUMN].astype(float)
    print("Overall metric summary:")
    print(f"mean: {vals_all.mean()}")
    print(f"std:  {vals_all.std()}")
    print(f"min:  {vals_all.min()}")
    print(f"max:  {vals_all.max()}")

    wt_metric = find_wt_metric(df)
    if wt_metric is not None:
        print(f"WT {METRIC_COLUMN}: {wt_metric}")
    else:
        print("WT not found (no WT_ID match and/or no single is_wt==1 row). Plotting without WT line.")

    steps = sorted(df[STEP_COLUMN].dropna().unique())
    print(f"Found mutation steps: {steps}")

    for step in steps:
        sub = df[df[STEP_COLUMN] == step]
        values = sub[METRIC_COLUMN].astype(float)

        outfile = OUTDIR / f"{METRIC_COLUMN}_step{int(step)}.png"
        print(f"Plotting step={step} | N={len(sub)} -> {outfile}")

        plot_step(values, step=step, wt_metric=wt_metric, outfile=outfile)

    print(f"\nSaved all step plots in: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()