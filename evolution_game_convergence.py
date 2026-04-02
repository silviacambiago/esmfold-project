"""
Evolution walk using ESM2 scoring with CONVERGENCE-BASED STOPPING

Stops when score stabilizes instead of fixed number of steps.

Convergence criteria:
1. Best score hasn't improved for PATIENCE_STEPS
2. Score variance in recent window below VARIANCE_THRESHOLD
3. Maximum steps reached (safety limit)

Runs for ALL wild types found in wild_types/ directory.
"""

from pathlib import Path
import argparse
import gzip
import random
import time
from typing import Tuple, List, Optional
from collections import deque

import numpy as np
import pandas as pd
import torch
import esm
import matplotlib.pyplot as plt

WT_DIR = "wild_types"
OUTDIR = "evolution_walk_out_total_random"

ESM2_MODEL = "esm2_t33_650M_UR50D"
DEVICE = "cuda"

SEED_INIT   = 42    # seeds initial random sequence generation
SEED_MUTATE = 1000  # seeds the mutation/proposal process

# Convergence parameters
MAX_STEPS = 50000                     # Safety limit (won't run forever)
PATIENCE_STEPS = 1000                 # Stop if no improvement for this many steps
VARIANCE_THRESHOLD = 1e-10            # Stop if variance in recent window below this
VARIANCE_WINDOW = 100                 # Window size for variance calculation
MIN_STEPS = 1000                      # Minimum steps before allowing convergence

PROPOSALS_PER_STEP = 100
MOVE_RULE = "non_negative"
PRINT_EVERY = 50

BIAS_TOWARDS_WT = False
BIAS_PROB = 0.30

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_SINGLE_PROTEIN = False  # set to True by main() when --protein is used


def read_fasta_one(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            lines = f.read().strip().splitlines()
    else:
        lines = path.read_text().strip().splitlines()
    return "".join(ln.strip() for ln in lines if not ln.startswith(">")).replace(" ", "").upper()


def mutate_one(seq: str, pos: int, aa: str) -> str:
    s = list(seq)
    s[pos] = aa
    return "".join(s)


def load_esm2(model_name: str, device: str):
    if model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        raise ValueError("Unsupported ESM2 model name")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.eval().to(dev)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, dev


@torch.no_grad()
def score_batch_fast(seqs: List[str], model, batch_converter, device) -> List[float]:
    """BATCHED fast proxy score - processes all sequences in one GPU pass."""
    if not seqs:
        return []

    batch = [(f"seq_{i}", seq) for i, seq in enumerate(seqs)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[], return_contacts=False)
    logits = out["logits"]
    log_probs = logits.log_softmax(dim=-1)

    scores = []
    for i in range(len(seqs)):
        true_tokens = tokens[i, 1:-1]
        token_log_probs = log_probs[i, 1:-1, true_tokens]
        scores.append(float(token_log_probs.mean().cpu()))

    return scores


def propose_mutations(
    current: str,
    wt: Optional[str],
    k: int,
    rng: random.Random,
) -> List[Tuple[int, str, str]]:
    """Returns proposals [(pos, new_aa, new_seq), ...]"""
    L = len(current)
    props = []
    for _ in range(k):
        pos = rng.randrange(L)

        if wt is not None and BIAS_TOWARDS_WT and rng.random() < BIAS_PROB:
            aa = wt[pos]
        else:
            aa = rng.choice(AMINO_ACIDS)

        if aa == current[pos]:
            aa = rng.choice(AMINO_ACIDS)
            if aa == current[pos]:
                continue

        new_seq = mutate_one(current, pos, aa)
        props.append((pos, aa, new_seq))
    return props


def accept_move(rule: str, old: float, new: float, rng: random.Random) -> bool:
    return new >= old


def check_convergence(
    step: int,
    best_score: float,
    best_score_step: int,
    score_history: deque,
) -> Tuple[bool, str]:
    if step < MIN_STEPS:
        return False, ""

    steps_since_improvement = step - best_score_step
    if steps_since_improvement >= PATIENCE_STEPS:
        return True, f"No improvement for {PATIENCE_STEPS} steps"

    if len(score_history) >= VARIANCE_WINDOW:
        recent_scores = list(score_history)
        variance = np.var(recent_scores)
        if variance < VARIANCE_THRESHOLD:
            return True, f"Score variance ({variance:.2e}) below threshold ({VARIANCE_THRESHOLD:.2e})"

    return False, ""


def run_one(wt_fasta: Path, protein_name: str, model, batch_converter, device,
            rng_init: random.Random, rng_mutate: random.Random):
    print(f"\n{'='*70}", flush=True)
    print(f"  PROTEIN: {protein_name}", flush=True)
    print(f"  WT:      {wt_fasta}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t0 = time.perf_counter()

    # Single-protein mode (--protein): OUTDIR is the final output path.
    # Multi-protein mode: append protein_name so each protein gets its own subdir.
    outdir = Path(OUTDIR) if _SINGLE_PROTEIN else Path(OUTDIR) / protein_name
    outdir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta_one(wt_fasta)
    L = len(wt_seq)
    print(f"  WT length: {L}", flush=True)

    # Random start: rng_init determines the initial sequence.
    # Same seed_init → always the same starting point, regardless of mutations.
    current = "".join(rng_init.choice(AMINO_ACIDS) for _ in range(L))
    current_score = score_batch_fast([current], model, batch_converter, device)[0]
    best_seq, best_score = current, current_score
    best_score_step = 0

    print(f"  Initial score: {current_score:.6f}\n", flush=True)

    rows = [{
        "step": 0,
        "score": current_score,
        "best_score_so_far": best_score,
        "hamming_to_wt": sum(a != b for a, b in zip(current, wt_seq)),
        "pos": "",
        "aa": "",
        "accepted": 1,
        "sequence": current,
    }]

    step_times = []
    score_history = deque(maxlen=VARIANCE_WINDOW)
    score_history.append(current_score)

    step = 0
    converged = False
    convergence_reason = ""

    while step < MAX_STEPS and not converged:
        step += 1
        t_step = time.perf_counter()

        # rng_mutate drives all position/amino-acid choices during the walk.
        # Same seed_mutate → identical mutation sequence, regardless of starting point.
        props = propose_mutations(current, wt_seq, PROPOSALS_PER_STEP, rng_mutate)
        candidate_seqs = [seq for _, _, seq in props]
        scores = score_batch_fast(candidate_seqs, model, batch_converter, device)

        scored = [(scores[i], props[i][0], props[i][1], props[i][2])
                  for i in range(len(props))]
        scored.sort(key=lambda x: x[0], reverse=True)
        cand_score, cand_pos, cand_aa, cand_seq = scored[0]

        do_accept = accept_move(MOVE_RULE, current_score, cand_score, rng_mutate)
        if do_accept:
            current, current_score = cand_seq, cand_score

        if current_score > best_score:
            best_seq, best_score = current, current_score
            best_score_step = step

        score_history.append(current_score)

        rows.append({
            "step": step,
            "score": current_score,
            "best_score_so_far": best_score,
            "hamming_to_wt": sum(a != b for a, b in zip(current, wt_seq)),
            "pos": cand_pos,
            "aa": cand_aa,
            "accepted": int(do_accept),
            "sequence": current,
        })

        step_time = time.perf_counter() - t_step
        step_times.append(step_time)

        converged, convergence_reason = check_convergence(
            step, best_score, best_score_step, score_history
        )

        if step % PRINT_EVERY == 0 or converged or step == MAX_STEPS:
            avg_time = sum(step_times) / len(step_times)
            steps_since_best = step - best_score_step
            var_str = (f"var={np.var(list(score_history)):.2e}"
                       if len(score_history) >= VARIANCE_WINDOW else "var=N/A")
            print(f"  [{step:5d}] score={current_score:.4f} best={best_score:.4f} "
                  f"hamWT={rows[-1]['hamming_to_wt']:4d} "
                  f"| no_impr={steps_since_best:4d} {var_str} "
                  f"| {step_time:.2f}s/step", flush=True)

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "trajectory.csv", index=False)
    (outdir / "best.fasta").write_text(
        f">best_score={best_score:.6f}_step={best_score_step}\n{best_seq}\n"
    )
    with open(outdir / "convergence_info.txt", "w") as f:
        f.write(f"Protein: {protein_name}\n")
        f.write(f"Total steps: {step}\n")
        f.write(f"Converged: {converged}\n")
        f.write(f"Convergence reason: {convergence_reason}\n")
        f.write(f"Best score: {best_score:.6f}\n")
        f.write(f"Best score at step: {best_score_step}\n")
        if len(score_history) >= VARIANCE_WINDOW:
            f.write(f"Final variance: {np.var(list(score_history)):.6e}\n")

    # Plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(df["step"], df["score"], alpha=0.7, label="score")
    plt.plot(df["step"], df["best_score_so_far"], linewidth=2, label="best")
    if converged:
        plt.axvline(step, color="red", linestyle="--", alpha=0.5, label="converged")
    plt.axvline(best_score_step, color="green", linestyle="--", alpha=0.5, label="best found")
    plt.xlabel("Step"); plt.ylabel("ESM2 score"); plt.title(f"{protein_name} - Score")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(df["step"], df["hamming_to_wt"], color="purple", linewidth=1.5)
    if converged:
        plt.axvline(step, color="red", linestyle="--", alpha=0.5, label="converged")
    plt.axvline(best_score_step, color="green", linestyle="--", alpha=0.5, label="best score")
    plt.xlabel("Step"); plt.ylabel("Hamming to WT"); plt.title(f"{protein_name} - Divergence")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    if len(df) > VARIANCE_WINDOW:
        variances = [np.var(df["score"].iloc[i - VARIANCE_WINDOW:i])
                     for i in range(VARIANCE_WINDOW, len(df))]
        plt.plot(range(VARIANCE_WINDOW, len(df)), variances, color="orange", label="rolling variance")
        plt.axhline(VARIANCE_THRESHOLD, color="red", linestyle="--", label="threshold")
        plt.xlabel("Step"); plt.ylabel("Variance"); plt.title(f"{protein_name} - Convergence")
        plt.legend(); plt.grid(True, alpha=0.3); plt.yscale("log")

    plt.tight_layout()
    plt.savefig(outdir / "score_curve.png", dpi=300)
    plt.close()

    runtime = time.perf_counter() - t0
    stop_reason = f"CONVERGED ({convergence_reason})" if converged else "HIT MAX STEPS"
    print(f"\n  [{protein_name}] {stop_reason} | steps={step} | best={best_score:.6f} | {runtime/60:.1f}min\n",
          flush=True)


def main():
    global OUTDIR, _SINGLE_PROTEIN
    parser = argparse.ArgumentParser(description="Convergence-based evolution walk")
    parser.add_argument("--wt-dir",      default=WT_DIR,      help="Directory with wild-type FASTAs")
    parser.add_argument("--outdir",      default=OUTDIR,      help="Output directory")
    parser.add_argument("--seed-init",   type=int, default=SEED_INIT,
                        help="Seed for initial random sequence generation")
    parser.add_argument("--seed-mutate", type=int, default=SEED_MUTATE,
                        help="Seed for mutation/proposal process")
    parser.add_argument("--protein",     default=None,
                        help="Run only this protein (stem of FASTA filename, e.g. insulin_human_wt). "
                             "If omitted, all wild types in --wt-dir are run.")
    args = parser.parse_args()

    print("\n" + "="*70, flush=True)
    label = f"PROTEIN: {args.protein}" if args.protein else "ALL WILD TYPES"
    print(f"  CONVERGENCE-BASED EVOLUTION WALK — {label}", flush=True)
    print("="*70 + "\n", flush=True)

    wt_dir = Path(args.wt_dir)
    wt_files = sorted(wt_dir.glob("*.fasta")) + sorted(wt_dir.glob("*.fasta.gz"))
    if not wt_files:
        print(f"ERROR: No .fasta / .fasta.gz files found in {wt_dir.absolute()}", flush=True)
        return

    if args.protein:
        wt_files = [f for f in wt_files if f.name.startswith(args.protein)]
        if not wt_files:
            print(f"ERROR: No FASTA found matching --protein '{args.protein}' in {wt_dir.absolute()}", flush=True)
            return

    print(f"  Found {len(wt_files)} wild type(s) to run:", flush=True)
    for f in wt_files:
        print(f"    • {f.name}", flush=True)

    print(f"\n  Model:          {ESM2_MODEL}", flush=True)
    print(f"  Device:         {DEVICE}", flush=True)
    print(f"  Max steps:      {MAX_STEPS}", flush=True)
    print(f"  Proposals/step: {PROPOSALS_PER_STEP}", flush=True)
    print(f"  Seed init:      {args.seed_init}", flush=True)
    print(f"  Seed mutate:    {args.seed_mutate}", flush=True)
    print(f"  Output:         {Path(args.outdir).absolute()}", flush=True)
    print(f"\nLoading ESM2 model...", flush=True)

    model, alphabet, batch_converter, device = load_esm2(ESM2_MODEL, DEVICE)
    print("✓ Model loaded\n", flush=True)

    # ── Seeds ────────────────────────────────────────────────────────────────
    # Values come from --seed-init and --seed-mutate CLI arguments, which are
    # set by the SLURM script as: user_base_seed + SLURM_ARRAY_TASK_ID.
    # Two separate RNGs ensure init and mutation processes are independently
    # reproducible — changing one seed does not affect the other.
    rng_init   = random.Random(args.seed_init)   # controls: what is the starting sequence
    rng_mutate = random.Random(args.seed_mutate) # controls: which mutations are proposed
    np.random.seed(args.seed_init)

    # Override OUTDIR with CLI arg
    OUTDIR = args.outdir
    _SINGLE_PROTEIN = args.protein is not None

    for wt_fasta in wt_files:
        name = wt_fasta.name
        for suffix in (".fasta.gz", ".fasta"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        run_one(wt_fasta, name, model, batch_converter, device, rng_init, rng_mutate)

    print("\n" + "="*70, flush=True)
    print("  ALL PROTEINS DONE", flush=True)
    print("="*70 + "\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted\n", flush=True)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n", flush=True)
        raise
