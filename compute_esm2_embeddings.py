"""
Compute ESM2 embeddings and scores from esm2_analysis output.

Input:
    - CSV produced by esm2_analysis.py, with columns:
        base_id, variant_id, sequence, is_wt, length,
        log_pseudo_likelihood, avg_log_prob, pseudo_perplexity,
        n_mut_positions, llr_vs_wt, mut_positions

What this script does:
    - Uses one variant as wild-type (by ID or by is_wt==1)
    - Computes Hamming distance to WT for all sequences
    - Defines a functional proxy from an ESM2 metric (avg_log_prob)
      and automatically chooses a threshold based on the distribution
      of deviations from the WT metric (relative threshold).
    - Runs ESM2 again to:
        * compute mean embedding per sequence
        * compute token-level log-prob score (esm_score)
    - Saves:
        * .npz with embeddings + labels + distances
        * a human-readable CSV summary with all relevant columns
"""

from pathlib import Path
from io import StringIO
import time

import numpy as np
import pandas as pd
import torch
import esm

# CSV produced by esm2_analysis.py
CSV_PATH = "tests/esm2_ubiquitin_multi/esm2_scores.csv"

# variant_id of the WT sequence (as written in esm2_analysis output)
WT_ID = "ubiquitin_wt"

OUTPUT_NPZ = "esm2_ubiquitin_multi_embeddings.npz"
SUMMARY_CSV = "ubiquitin_multi_esm2_summary.csv"

DEVICE = "cuda"
ESM2_MODEL = "esm2_t36_3B_UR50D"

# column from esm2_analysis to use as ESM-based "fitness-like" metric
METRIC_COLUMN = "avg_log_prob"

# Fraction of mutants to consider "functional-like" (closest to WT in metric space)
FUNCTIONAL_FRACTION = 0.7

def compute_hamming_distance(seq, wt):
    assert len(seq) == len(wt), "All sequences must have same length as WT"
    return sum(a != b for a, b in zip(seq, wt))


def load_and_fix_csv(path: str) -> pd.DataFrame:
    """
    Load esm2_scores.csv and fix lines that have extra commas in mut_positions.

    Strategy:
      - Read file as raw text.
      - Split each line by ','.
      - If a line has more columns than the header, merge all trailing pieces
        into the last column (mut_positions) using commas again.
    """
    text = Path(path).read_text().rstrip("\n")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"CSV file {path} appears to be empty")

    header = lines[0]
    cols = header.split(",")
    n_cols = len(cols)

    fixed_lines = [header]
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= n_cols:
            fixed_lines.append(line)
        else:
            head = parts[: n_cols - 1]
            tail = ",".join(parts[n_cols - 1 :])
            fixed_line = ",".join(head + [tail])
            fixed_lines.append(fixed_line)

    fixed_csv = "\n".join(fixed_lines)
    df = pd.read_csv(StringIO(fixed_csv), sep=",")
    return df


def main():
    t0 = time.perf_counter()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading CSV from: {CSV_PATH}")
    df = load_and_fix_csv(CSV_PATH)

    if "variant_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV must contain at least 'variant_id' and 'sequence' columns")

    df["seq_id"] = df["variant_id"].astype(str)

    if WT_ID is not None:
        wt_row = df.loc[df["seq_id"] == WT_ID]
        if wt_row.empty:
            raise ValueError(f"WT_ID '{WT_ID}' not found in column 'variant_id'")
    else:
        if "is_wt" not in df.columns:
            raise ValueError("No WT_ID specified and no 'is_wt' column available.")
        wt_row = df.loc[df["is_wt"] == 1]
        if wt_row.empty:
            raise ValueError("No row with is_wt == 1 found in CSV")
        if len(wt_row) > 1:
            raise ValueError("Multiple rows with is_wt == 1; please set WT_ID explicitly.")

    wt_row = wt_row.iloc[0]
    wt_seq = wt_row["sequence"]
    wt_id_used = wt_row["seq_id"]

    print(f"WT seq_id: {wt_id_used} | length: {len(wt_seq)}")
    print(f"Total sequences in CSV: {len(df)}")

    df["hamming_distance"] = df["sequence"].apply(
        lambda s: compute_hamming_distance(s, wt_seq)
    )

    if METRIC_COLUMN not in df.columns:
        raise ValueError(f"Configured METRIC_COLUMN '{METRIC_COLUMN}' not found in CSV")

    df["fitness_proxy"] = df[METRIC_COLUMN].astype(float)

    wt_metric = float(wt_row[METRIC_COLUMN])

    delta = wt_metric - df["fitness_proxy"]

    is_wt_mask = (df["seq_id"] == wt_id_used)
    delta_nonwt = delta[~is_wt_mask].values

    if len(delta_nonwt) == 0:
        raise RuntimeError("No non-WT sequences found to define a functional threshold.")

    delta_nonwt_clipped = np.clip(delta_nonwt, a_min=0.0, a_max=None)

    q = float(FUNCTIONAL_FRACTION)
    if not (0.0 < q < 1.0):
        raise ValueError("FUNCTIONAL_FRACTION must be in (0, 1).")

    delta_threshold = np.quantile(delta_nonwt_clipped, q)

    functional_threshold = wt_metric - delta_threshold

    df["functional_label"] = (df["fitness_proxy"] >= functional_threshold).astype(int)

    n_func = int(df["functional_label"].sum())
    n_nonfunc = int(len(df) - n_func)

    print(f"\nUsing metric column '{METRIC_COLUMN}' as functional proxy")
    print(f"WT {METRIC_COLUMN}: {wt_metric:.4f}")
    print(f"FUNCTIONAL_FRACTION (≈ fraction of mutants considered functional): {FUNCTIONAL_FRACTION:.2f}")
    print(f"Δ-threshold (WT_metric - metric): {delta_threshold:.4f}")
    print(f"Functional threshold (metric >=): {functional_threshold:.4f}")
    print(f"Functional sequences: {n_func}")
    print(f"Non-functional sequences: {n_nonfunc}")

    print(f"\nLoading ESM2 model: {ESM2_MODEL}")
    if ESM2_MODEL == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        repr_layer = 36
    elif ESM2_MODEL == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        repr_layer = 33
    else:
        raise ValueError("ESM2_MODEL must be 'esm2_t36_3B_UR50D' or 'esm2_t33_650M_UR50D'")

    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    seq_ids = []
    embeddings = []
    esm_scores = []
    fitness_vals = []
    labels = []
    hamdist = []

    print("\nStarting ESM2 forward passes...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            sid = row["seq_id"]
            seq = row["sequence"]

            batch = [(sid, seq)]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)

            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)

            reps = out["representations"][repr_layer]
            reps = reps[0, 1:-1, :]
            rep_mean = reps.mean(dim=0).cpu().numpy()

            logits = out["logits"]
            log_probs = logits.log_softmax(dim=-1)
            true_tokens = tokens[0, 1:-1]
            token_log_probs = log_probs[0, 1:-1, true_tokens]
            score = float(token_log_probs.mean().cpu())

            seq_ids.append(sid)
            embeddings.append(rep_mean)
            esm_scores.append(score)
            fitness_vals.append(float(row["fitness_proxy"]))
            labels.append(int(row["functional_label"]))
            hamdist.append(int(row["hamming_distance"]))

            if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
                print(f"[INFO] Processed {idx+1}/{len(df)} sequences")

    embeddings = np.stack(embeddings, axis=0)
    esm_scores = np.array(esm_scores)
    fitness_vals = np.array(fitness_vals)
    labels = np.array(labels)
    hamdist = np.array(hamdist)
    seq_ids = np.array(seq_ids, dtype=object)

    print(f"Embeddings shape: {embeddings.shape}")

    out_path = Path(OUTPUT_NPZ)
    np.savez(
        out_path,
        seq_ids=seq_ids,
        embeddings=embeddings,
        esm_scores=esm_scores,
        fitness=fitness_vals,
        labels=labels,
        hamming_distance=hamdist,
        wt_id=wt_id_used,
        fitness_threshold=functional_threshold,
        model=ESM2_MODEL,
        metric_column=METRIC_COLUMN,
        wt_metric=wt_metric,
        delta_threshold=delta_threshold,
        functional_fraction=FUNCTIONAL_FRACTION,
    )
    print(f"Saved embeddings and metadata to: {out_path.resolve()}")

    summary_cols = [
        "seq_id",
        "base_id",
        "variant_id",
        "sequence",
        "is_wt",
        "length",
        "log_pseudo_likelihood",
        "avg_log_prob",
        "pseudo_perplexity",
        "n_mut_positions",
        "llr_vs_wt",
        "mut_positions",
        "fitness_proxy",
        "functional_label",
        "hamming_distance",
    ]

    available_cols = [c for c in summary_cols if c in df.columns]
    summary_df = df[available_cols].copy()
    summary_df["esm_score"] = esm_scores
    summary_df["wt_metric"] = wt_metric
    summary_df["functional_threshold"] = functional_threshold
    summary_df["delta_threshold"] = delta_threshold
    summary_df["functional_fraction"] = FUNCTIONAL_FRACTION

    summary_path = Path(SUMMARY_CSV)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path.resolve()}")

    print("Preview of summary:")
    print(summary_df.head().to_string(index=False))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    minutes, seconds = divmod(elapsed, 60)
    print(f"\nTotal runtime: {elapsed:.2f} s (~{int(minutes)} min {seconds:4.1f} s)")


if __name__ == "__main__":
    main()
