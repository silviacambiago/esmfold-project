#!/usr/bin/env python3
"""
Analyze evolution walk results produced by evolution_game_convergence.py.

Auto-detects directory layout:

  Depth-1, multi-seed (one protein, many seeds):
    batch_dir/run_XX_seedYY/trajectory.csv[.gz]

  Depth-1, multi-protein (many proteins, one seed each):
    batch_dir/protein_name/trajectory.csv[.gz]

  Depth-2 (many proteins, many seeds):
    batch_dir/run_label/protein_name/trajectory.csv[.gz]

  Depth-3 (HPC artifact: outer_dir/run_label/protein_name/trajectory.csv):
    batch_dir/outer/run_label/protein_name/trajectory.csv[.gz]
    The outer directory is ignored; protein and run_label are taken from
    the two innermost components.

Protein identity is read from convergence_info.txt[.gz] when present,
otherwise inferred from sequence length (depth-1) or directory name.

Usage:
    python analyze_batch_results.py --batch-dir evolution_convergence_results
    python analyze_batch_results.py --batch-dir evolution_walk_out_total_random --outdir plots/
"""

import argparse
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_trajectory(path: Path) -> pd.DataFrame:
    with _open(path) as f:
        return pd.read_csv(f)


def read_convergence_info(run_dir: Path) -> dict:
    for name in ("convergence_info.txt.gz", "convergence_info.txt"):
        p = run_dir / name
        if p.exists():
            with _open(p) as f:
                return {k.strip(): v.strip()
                        for line in f if ":" in line
                        for k, v in [line.split(":", 1)]}
    return {}


def find_runs(batch_dir: Path):
    """Return list of {protein, run_label, df} dicts."""
    trajs = sorted(batch_dir.rglob("trajectory.csv")) + \
            sorted(batch_dir.rglob("trajectory.csv.gz"))
    # de-duplicate: prefer .csv over .csv.gz when both exist in same dir
    seen, unique = set(), []
    for t in trajs:
        key = t.parent
        if key not in seen:
            seen.add(key)
            unique.append(t)

    records = []
    for traj in unique:
        rel   = traj.relative_to(batch_dir)
        parts = rel.parts
        try:
            df = read_trajectory(traj)
        except Exception as e:
            print(f"  Warning: skipping {traj}: {e}")
            continue

        info = read_convergence_info(traj.parent)

        if len(parts) == 2:        # depth-1: dir/trajectory.csv
            dir_name = parts[0]
            protein  = info.get("Protein")
            if not protein:
                seq_len = len(df["sequence"].iloc[0]) if "sequence" in df.columns else "?"
                protein = f"protein_len{seq_len}"
            records.append({"protein": protein, "run_label": dir_name, "df": df})

        elif len(parts) == 3:      # depth-2: run_label/protein/trajectory.csv
            run_label = parts[0]
            protein   = info.get("Protein", parts[1])
            records.append({"protein": protein, "run_label": run_label, "df": df})

        elif len(parts) == 4:      # depth-3: outer/run_label/protein/trajectory.csv
            run_label = parts[1]   # outer dir (parts[0]) is an artifact, ignored
            protein   = info.get("Protein", parts[2])
            records.append({"protein": protein, "run_label": run_label, "df": df})

    return records


def pad(arrays):
    """Pad shorter 1-D arrays to max length by repeating last value."""
    L = max(len(a) for a in arrays)
    return np.array([np.pad(a, (0, L - len(a)), mode="edge") for a in arrays])


def plot_multi_seed(protein, runs, outdir):
    dfs   = [r["df"] for r in runs]
    n     = len(runs)
    best  = pad([df["best_score_so_far"].values for df in dfs])
    hamm  = pad([df["hamming_to_wt"].values     for df in dfs])
    steps = np.arange(best.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{protein}  —  {n} runs", fontsize=13)

    def _overlay(ax, mat, color, ylabel, title):
        for row in mat:
            ax.plot(steps, row, alpha=0.25, linewidth=0.7, color=color)
        m, s = mat.mean(0), mat.std(0)
        ax.plot(steps, m, "r-", linewidth=2, label="mean", zorder=5)
        ax.fill_between(steps, m - s, m + s, color="red", alpha=0.15, label="±1 std")
        ax.set_xlabel("Step"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _overlay(axes[0], best, "steelblue",    "Best ESM2 score (avg log-prob)", "Score trajectories")
    _overlay(axes[1], hamm, "mediumorchid", "Hamming distance to WT",         "Divergence from WT")

    finals  = [df["best_score_so_far"].iloc[-1] for df in dfs]
    n_steps = [len(df) - 1                      for df in dfs]
    ax = axes[2]
    ax.hist(finals, bins=min(20, n), edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(finals),   color="red",    linestyle="--", linewidth=1.5,
               label=f"mean   {np.mean(finals):.4f}")
    ax.axvline(np.median(finals), color="orange", linestyle="--", linewidth=1.5,
               label=f"median {np.median(finals):.4f}")
    ax.set_xlabel("Final best score"); ax.set_ylabel("Count")
    ax.set_title(f"Score distribution\nconv. steps: {np.mean(n_steps):.0f}±{np.std(n_steps):.0f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / f"{protein.replace('/', '_')}_runs.png", dpi=150)
    plt.close()
    return finals, n_steps


def plot_cross_protein(summaries, outdir):
    proteins = sorted(summaries)
    n        = len(proteins)

    fig, axes = plt.subplots(1, 3, figsize=(max(12, n * 2), 6))
    fig.suptitle(f"Cross-protein comparison  ({n} proteins)", fontsize=13)

    def _panel(ax, data, ylabel, title, color):
        single = all(len(d) == 1 for d in data)
        if single:
            ax.bar(range(n), [d[0] for d in data],
                   color=color, edgecolor="black", alpha=0.8)
            ax.set_xticks(range(n))
        else:
            bp = ax.boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.6))
            ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(proteins, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3, axis="y")

    _panel(axes[0], [summaries[p]["finals"]        for p in proteins],
           "Best ESM2 score (avg log-prob)", "Best score per protein",    "steelblue")
    _panel(axes[1], [summaries[p]["n_steps"]       for p in proteins],
           "Steps to convergence",           "Convergence speed",          "coral")
    _panel(axes[2], [summaries[p]["final_hamming"] for p in proteins],
           "Hamming distance to WT",         "Final divergence from WT",   "mediumseagreen")

    plt.tight_layout()
    plt.savefig(outdir / "cross_protein_comparison.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze evolution walk results")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--outdir",    default=None,
                        help="Output directory (default: same as --batch-dir)")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    outdir    = Path(args.outdir) if args.outdir else batch_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {batch_dir} ...")
    records = find_runs(batch_dir)
    if not records:
        print("No trajectory files found."); return

    by_protein = defaultdict(list)
    for r in records:
        by_protein[r["protein"]].append(r)

    print(f"Found {len(records)} runs across {len(by_protein)} protein(s)\n")

    summary_rows, protein_summaries = [], {}

    for protein, runs in sorted(by_protein.items()):
        print(f"  {protein}: {len(runs)} run(s)")
        finals, n_steps = plot_multi_seed(protein, runs, outdir)
        final_hamming   = [r["df"]["hamming_to_wt"].iloc[-1] for r in runs]

        protein_summaries[protein] = {
            "finals": finals, "n_steps": n_steps, "final_hamming": final_hamming
        }

        best_run = max(runs, key=lambda r: r["df"]["best_score_so_far"].iloc[-1])
        best_df  = best_run["df"]
        best_idx = best_df["best_score_so_far"].idxmax()
        safe     = protein.replace("/", "_").replace(" ", "_")
        (outdir / f"{safe}_best.fasta").write_text(
            f">{protein}  score={best_df['best_score_so_far'].iloc[-1]:.6f}\n"
            f"{best_df.loc[best_idx, 'sequence']}\n"
        )

        for r, sc, st, hm in zip(runs, finals, n_steps, final_hamming):
            summary_rows.append({
                "protein": protein, "run_label": r["run_label"],
                "best_score": sc, "n_steps": st, "final_hamming_wt": hm,
                "final_sequence": r["df"]["sequence"].iloc[-1],
            })

    if len(by_protein) > 1:
        plot_cross_protein(protein_summaries, outdir)

    pd.DataFrame(summary_rows) \
      .sort_values(["protein", "best_score"], ascending=[True, False]) \
      .to_csv(outdir / "summary.csv", index=False)

    print(f"\n{'protein':<30} {'runs':>5} {'best':>9} {'mean':>9} {'std':>7} {'steps(mean)':>12}")
    print("-" * 76)
    for prot, ps in sorted(protein_summaries.items()):
        f, s = ps["finals"], ps["n_steps"]
        print(f"  {prot:<28} {len(f):>5} {max(f):>9.4f} {np.mean(f):>9.4f} "
              f"{np.std(f):>7.4f} {np.mean(s):>10.0f}±{np.std(s):.0f}")

    print(f"\nOutputs written to: {outdir.absolute()}")


if __name__ == "__main__":
    main()
