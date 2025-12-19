"""
Evolution walk using ESM2 scoring.

Outputs:
- outdir/trajectory.csv : per-step sequence + score
- outdir/best.fasta     : best sequence found
- outdir/score_curve.png
"""

from pathlib import Path
import math
import random
import time
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import esm
import matplotlib.pyplot as plt

WT_FASTA = "insulin_fastas/insulin_human_wt.fasta"
OUTDIR = "evolution_walk_out"

ESM2_MODEL = "esm2_t33_650M_UR50D"
DEVICE = "cuda"

SEED = 42

N_STEPS = 400                        # length of the walk
PROPOSALS_PER_STEP = 30              # how many candidate single mutations per step

# Mutation proposal policy
# - "non_negative": take best only if it doesn't decrease score, else stay
MOVE_RULE = "non_negative"

USE_TRUE_PLL = True

# Alphabet (ESM standard)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

BIAS_TOWARDS_WT = True
BIAS_PROB = 0.30                     # probability that a proposal uses WT aa at that pos

def read_fasta_one(path: str) -> str:
    txt = Path(path).read_text().strip().splitlines()
    seq = "".join([ln.strip() for ln in txt if not ln.startswith(">")]).replace(" ", "").upper()
    return seq


def mutate_one(seq: str, pos: int, aa: str) -> str:
    s = list(seq)
    s[pos] = aa
    return "".join(s)


def load_esm2(model_name: str, device: str):
    if model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        repr_layer = 36
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        repr_layer = 33
    else:
        raise ValueError("Unsupported ESM2 model name")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.eval().to(dev)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, dev, repr_layer


@torch.no_grad()
def score_fast_mean_logprob(seq: str, model, batch_converter, device) -> float:
    """
    FAST proxy score:
      mean over positions of log p(x_i | unmasked forward logits)

    Note: This is NOT true PLL, but it's cheap and stable for search.
    """
    batch = [("seq", seq)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[], return_contacts=False)
    logits = out["logits"]            # (1, L+2, vocab)
    log_probs = logits.log_softmax(dim=-1)

    true_tokens = tokens[0, 1:-1]     # (L,)
    token_log_probs = log_probs[0, 1:-1, true_tokens]  # (L,)
    return float(token_log_probs.mean().cpu())


@torch.no_grad()
def score_true_pll(seq: str, model, alphabet, batch_converter, device, pos_batch_size: int = 16) -> float:
    """
    SLOW avPLL:
      PLL = sum_i log p(x_i | x_-i) via masking each position
      return avPLL = PLL / L
    """
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    mask_idx = alphabet.mask_idx
    L = len(seq)
    pll = 0.0


    for start in range(0, L, pos_batch_size):
        end = min(L, start + pos_batch_size)
        positions = list(range(start, end))
        bsz = len(positions)

        batch_tokens = tokens.repeat(bsz, 1)
        for r, pos in enumerate(positions):
            batch_tokens[r, pos + 1] = mask_idx

        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = out["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)

        for r, pos in enumerate(positions):
            true_idx = tokens[0, pos + 1]
            pll += log_probs[r, pos + 1, true_idx].item()

    return float(pll / L)


def propose_mutations(
    current: str,
    wt: Optional[str],
    k: int,
    rng: random.Random,
) -> List[Tuple[int, str, str]]:
    """
    Returns proposals [(pos, new_aa, new_seq), ...]
    """
    L = len(current)
    props = []
    for _ in range(k):
        pos = rng.randrange(L)

        if wt is not None and BIAS_TOWARDS_WT and rng.random() < BIAS_PROB:
            aa = wt[pos]
        else:
            aa = rng.choice(AMINO_ACIDS)

        if aa == current[pos]:
            # try once more quickly
            aa = rng.choice(AMINO_ACIDS)
            if aa == current[pos]:
                continue

        new_seq = mutate_one(current, pos, aa)
        props.append((pos, aa, new_seq))
    return props


def accept_move(rule: str, old: float, new: float, rng: random.Random) -> bool:
    return new >= old


def main():
    t0 = time.perf_counter()
    rng = random.Random(SEED)
    np.random.seed(SEED)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta_one(WT_FASTA)
    L = len(wt_seq)

    print(f"WT length: {L}")
    print(f"Model: {ESM2_MODEL} | DEVICE={DEVICE}")
    print(f"Steps: {N_STEPS} | proposals/step: {PROPOSALS_PER_STEP}")
    print(f"MOVE_RULE={MOVE_RULE} | USE_TRUE_PLL={USE_TRUE_PLL}")

    model, alphabet, batch_converter, device, _ = load_esm2(ESM2_MODEL, DEVICE)

    # start from random sequence
    current = "".join(rng.choice(AMINO_ACIDS) for _ in range(L))

    # scoring function
    if USE_TRUE_PLL:
        def score_fn(s: str) -> float:
            return score_true_pll(s, model, alphabet, batch_converter, device)
    else:
        def score_fn(s: str) -> float:
            return score_fast_mean_logprob(s, model, batch_converter, device)

    current_score = score_fn(current)
    best_seq, best_score = current, current_score

    rows = []
    rows.append({
        "step": 0,
        "score": current_score,
        "best_score_so_far": best_score,
        "hamming_to_wt": sum(a != b for a, b in zip(current, wt_seq)),
        "pos": "",
        "aa": "",
        "accepted": 1,
        "sequence": current,
    })

    for step in range(1, N_STEPS + 1):
        props = propose_mutations(current, wt_seq, PROPOSALS_PER_STEP, rng)

        # evaluate proposals
        scored = []
        for pos, aa, seq in props:
            s = score_fn(seq)
            scored.append((s, pos, aa, seq))

        # best candidate this step
        scored.sort(key=lambda x: x[0], reverse=True)
        cand_score, cand_pos, cand_aa, cand_seq = scored[0]

        do_accept = accept_move(MOVE_RULE, current_score, cand_score, rng)
        if do_accept:
            current, current_score = cand_seq, cand_score

        if current_score > best_score:
            best_seq, best_score = current, current_score

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

        if step % 50 == 0 or step == N_STEPS:
            print(f"[{step:4d}/{N_STEPS}] score={current_score:.4f} best={best_score:.4f} hamWT={rows[-1]['hamming_to_wt']}")

    # save trajectory
    df = pd.DataFrame(rows)
    traj_csv = outdir / "trajectory.csv"
    df.to_csv(traj_csv, index=False)
    print(f"Saved: {traj_csv}")

    # save best fasta
    best_fa = outdir / "best.fasta"
    best_fa.write_text(f">best_score={best_score:.6f}\n{best_seq}\n")
    print(f"Saved: {best_fa}")

    # plot score curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["score"], label="score")
    plt.plot(df["step"], df["best_score_so_far"], label="best so far")
    plt.xlabel("Step")
    plt.ylabel("ESM2 score")
    plt.title("Evolution walk score over time")
    plt.legend()
    plt.tight_layout()
    png = outdir / "score_curve.png"
    plt.savefig(png, dpi=300)
    plt.show()
    print(f"Saved: {png}")

    t1 = time.perf_counter()
    print(f"Done. Runtime: {(t1 - t0)/60:.2f} min")


if __name__ == "__main__":
    main()