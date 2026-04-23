"""
Evolution walk: Two-phase convergence toward target protein.

PHASE 1 — APPROACH (pure EDS, greedy):
  Start from random sequence. Optimize purely for EDS (embedding distance
  to WT). No PLL involved. Greedy: only accept mutations that improve EDS
  (move closer to WT in embedding space). This phase answers: "does EDS
  have a useful gradient from Hamming ~100 toward insulin?"
  Stops when EDS plateaus or Hamming target is reached.

PHASE 2 — REFINE (PLL + lambda*EDS, MCMC):
  Take the best sequence from Phase 1. Now optimize for protein quality
  (PLL) while maintaining structural proximity (EDS). Uses MCMC to escape
  local optima. This phase answers: "once close to insulin, can we find
  variants that are both insulin-like and high-quality?"

Additionally tests a CONTROL mode: pure PLL from random (no Phase 1),
to compare against the two-phase approach.

Outputs:
- outdir/trajectory.csv
- outdir/summary.csv
- outdir/best_*.fasta
- outdir/plots_phases.png      : 2x2 phase transition analysis
- outdir/plots_comparison.png  : 2x2 two-phase vs control
- outdir/plots_convergence.png : convergence diagnostics
- outdir/plots_eds.png         : EDS analysis
"""
import os
import warnings
from pathlib import Path
import random
import math
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*libomp.*")

import numpy as np
import pandas as pd
import torch
import esm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


WT_FASTA = "insulin_fastas/insulin_human_wt.fasta"
OUTDIR = "evolution_walk_v5_twophase"

REPR_LAYER = 33
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

N_RUNS = 5

# Phase 1: Pure EDS approach
P1_MAX_STEPS = 500
P1_PROPOSALS = 30
P1_PATIENCE = 100                    # stop if no EDS improvement for this many steps
P1_HAMMING_TARGET = 20               # switch to Phase 2 if reached

# Phase 2: PLL + EDS refinement
P2_MAX_STEPS = 500
P2_PROPOSALS = 30
P2_TOP_K = 3
P2_EDS_LAMBDA = 0.05               
P2_INITIAL_TEMP = 0.1
P2_MIN_TEMP = 0.001
P2_COOLING_RATE = 0.993
P2_PATIENCE = 150
P2_CONVERGENCE_WINDOW = 50
P2_VAR_THRESHOLD = 1e-6

# Control: pure PLL from random (same budget as Phase 1 + Phase 2)
CTRL_MAX_STEPS = 1000
CTRL_INITIAL_TEMP = 0.1

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

C_TWOPHASE = "#2ecc71"
C_CONTROL = "#e74c3c"
C_PHASE1 = "#3498db"
C_PHASE2 = "#e67e22"


def read_fasta_one(path):
    txt = Path(path).read_text().strip().splitlines()
    return "".join(ln.strip() for ln in txt if not ln.startswith(">")).replace(" ", "").upper()

def mutate_one(seq, pos, aa):
    s = list(seq); s[pos] = aa; return "".join(s)

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def load_esm2(device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(torch.device(device))
    return model, alphabet, alphabet.get_batch_converter()

def random_sequence(length, rng):
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


@torch.no_grad()
def score_true_pll(seq, model, alphabet, bc, device, pos_batch_size=16):
    _, _, tokens = bc([("seq", seq)]); tokens = tokens.to(device)
    mask_idx = alphabet.mask_idx; L = len(seq); pll = 0.0
    for start in range(0, L, pos_batch_size):
        end = min(L, start + pos_batch_size)
        positions = list(range(start, end)); bsz = len(positions)
        bt = tokens.repeat(bsz, 1)
        for r, pos in enumerate(positions): bt[r, pos + 1] = mask_idx
        out = model(bt, repr_layers=[], return_contacts=False)
        lp = torch.log_softmax(out["logits"], dim=-1)
        for r, pos in enumerate(positions):
            pll += lp[r, pos + 1, tokens[0, pos + 1]].item()
    return float(pll / L)

@torch.no_grad()
def compute_mllr(seq, pos, new_aa, model, alphabet, bc, device):
    _, _, tokens = bc([("seq", seq)]); tokens = tokens.to(device)
    tokens[0, pos + 1] = alphabet.mask_idx
    out = model(tokens, repr_layers=[], return_contacts=False)
    lp = torch.log_softmax(out["logits"], dim=-1)
    return float(lp[0, pos+1, alphabet.get_idx(new_aa)].item() -
                 lp[0, pos+1, alphabet.get_idx(seq[pos])].item())

@torch.no_grad()
def compute_entropy(seq, model, alphabet, bc, device, pos_batch_size=16):
    _, _, tokens = bc([("seq", seq)]); tokens = tokens.to(device)
    mask_idx = alphabet.mask_idx; L = len(seq)
    aa_idx = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    ent = np.zeros(L)
    for start in range(0, L, pos_batch_size):
        end = min(L, start + pos_batch_size)
        positions = list(range(start, end)); bsz = len(positions)
        bt = tokens.repeat(bsz, 1)
        for r, pos in enumerate(positions): bt[r, pos + 1] = mask_idx
        out = model(bt, repr_layers=[], return_contacts=False)
        lp = torch.log_softmax(out["logits"], dim=-1)
        for r, pos in enumerate(positions):
            p = lp[r, pos+1, aa_idx].exp()
            ent[pos] = -(p * lp[r, pos+1, aa_idx]).sum().item()
    return ent

@torch.no_grad()
def get_embeddings(seq, model, bc, device):
    _, _, tokens = bc([("seq", seq)]); tokens = tokens.to(device)
    out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    return out["representations"][REPR_LAYER][0, 1:-1, :]

def compute_eds(mut_emb, wt_emb):
    return -torch.norm(mut_emb - wt_emb).item()


def propose_mutations(current, k, rng, entropies=None):
    L = len(current)
    if entropies is not None:
        w = np.clip(entropies, 1e-8, None); w /= w.sum()
    else:
        w = [1.0/L] * L
    props = []
    for _ in range(k * 2):
        if len(props) >= k: break
        pos = rng.choices(range(L), weights=w, k=1)[0]
        aa = rng.choice(AMINO_ACIDS)
        if aa == current[pos]:
            aa = rng.choice(AMINO_ACIDS)
            if aa == current[pos]: continue
        props.append((pos, aa, mutate_one(current, pos, aa)))
    return props

def mcmc_accept(old_score, new_score, temperature, rng):
    delta = new_score - old_score
    if delta >= 0: return True
    if temperature <= 0: return False
    return rng.random() < math.exp(delta / temperature)


def run_phase1(start_seq, wt_seq, wt_emb, model, alphabet, bc, device, rng):
    current = start_seq
    cur_emb = get_embeddings(current, model, bc, device)
    current_eds = compute_eds(cur_emb, wt_emb)
    best_seq, best_eds = current, current_eds
    steps_since_imp = 0

    trajectory = [{
        "step": 0, "phase": 1,
        "eds": current_eds, "best_eds": best_eds,
        "hamming_to_wt": hamming(current, wt_seq),
        "true_pll": None,  # not computed in Phase 1
        "accepted": True,
    }]

    for step in range(1, P1_MAX_STEPS + 1):
        props = propose_mutations(current, P1_PROPOSALS, rng)
        if not props:
            steps_since_imp += 1
            trajectory.append({
                "step": step, "phase": 1,
                "eds": current_eds, "best_eds": best_eds,
                "hamming_to_wt": hamming(current, wt_seq),
                "true_pll": None, "accepted": False,
            })
            if steps_since_imp >= P1_PATIENCE: break
            continue

        best_cand_eds = -float('inf')
        best_cand_seq = None

        for pos, aa, seq in props:
            cemb = get_embeddings(seq, model, bc, device)
            ceds = compute_eds(cemb, wt_emb)
            if ceds > best_cand_eds:
                best_cand_eds = ceds
                best_cand_seq = seq

        accepted = False
        if best_cand_seq is not None and best_cand_eds > current_eds:
            current = best_cand_seq
            current_eds = best_cand_eds
            accepted = True

        if current_eds > best_eds:
            best_seq, best_eds = current, current_eds
            steps_since_imp = 0
        else:
            steps_since_imp += 1

        cur_ham = hamming(current, wt_seq)
        trajectory.append({
            "step": step, "phase": 1,
            "eds": current_eds, "best_eds": best_eds,
            "hamming_to_wt": cur_ham,
            "true_pll": None, "accepted": accepted,
        })

        if steps_since_imp >= P1_PATIENCE: break
        if cur_ham <= P1_HAMMING_TARGET:
            print(f"    Phase 1: Hamming target {P1_HAMMING_TARGET} reached at step {step}!")
            break

    n_steps = len(trajectory) - 1
    final_ham = hamming(best_seq, wt_seq)
    print(f"    Phase 1 done: {n_steps} steps | "
          f"EDS: {trajectory[0]['eds']:.1f} -> {best_eds:.1f} | "
          f"Ham: {trajectory[0]['hamming_to_wt']} -> {final_ham}")

    return trajectory, best_seq, best_eds


def run_phase2(start_seq, wt_seq, wt_emb, model, alphabet, bc, device, rng,
               step_offset=0):
    current = start_seq
    current_pll = score_true_pll(current, model, alphabet, bc, device)
    cur_emb = get_embeddings(current, model, bc, device)
    current_eds = compute_eds(cur_emb, wt_emb)

    eds_lambda = P2_EDS_LAMBDA
    current_score = current_pll + eds_lambda * current_eds
    best_seq, best_pll, best_eds = current, current_pll, current_eds
    best_score = current_score
    best_score_history = [best_score]
    steps_since_imp = 0
    accepted_up = 0; accepted_down = 0
    temp = P2_INITIAL_TEMP

    entropies = compute_entropy(current, model, alphabet, bc, device)

    trajectory = [{
        "step": step_offset, "phase": 2,
        "true_pll": current_pll, "best_pll": best_pll,
        "eds": current_eds, "best_eds": best_eds,
        "score": current_score, "best_score": best_score,
        "hamming_to_wt": hamming(current, wt_seq),
        "temperature": temp, "accepted": True, "accepted_downhill": False,
    }]

    for step in range(1, P2_MAX_STEPS + 1):
        global_step = step_offset + step

        props = propose_mutations(current, P2_PROPOSALS, rng, entropies)
        if not props:
            best_score_history.append(best_score)
            steps_since_imp += 1
            trajectory.append({
                "step": global_step, "phase": 2,
                "true_pll": current_pll, "best_pll": best_pll,
                "eds": current_eds, "best_eds": best_eds,
                "score": current_score, "best_score": best_score,
                "hamming_to_wt": hamming(current, wt_seq),
                "temperature": temp, "accepted": False, "accepted_downhill": False,
            })
            temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
            if steps_since_imp >= P2_PATIENCE: break
            continue

        # MLLR pre-filter
        mllr = [(compute_mllr(current, p, a, model, alphabet, bc, device), p, a, s)
                for p, a, s in props]
        mllr.sort(key=lambda x: x[0], reverse=True)

        best_cand_score = -float('inf')
        best_cand_seq = None
        best_cand_pll = -float('inf')
        best_cand_eds = current_eds

        for m, pos, aa, seq in mllr[:P2_TOP_K]:
            cpll = score_true_pll(seq, model, alphabet, bc, device)
            cemb = get_embeddings(seq, model, bc, device)
            ceds = compute_eds(cemb, wt_emb)
            cscore = cpll + eds_lambda * ceds

            if cscore > best_cand_score:
                best_cand_score = cscore
                best_cand_seq = seq
                best_cand_pll = cpll
                best_cand_eds = ceds

        # MCMC accept
        accepted = False; accepted_downhill = False
        if best_cand_seq is not None:
            if mcmc_accept(current_score, best_cand_score, temp, rng):
                accepted_downhill = (best_cand_score < current_score)
                current = best_cand_seq
                current_pll = best_cand_pll
                current_eds = best_cand_eds
                current_score = best_cand_score
                accepted = True
                if accepted_downhill: accepted_down += 1
                else: accepted_up += 1

        if current_score > best_score:
            best_seq, best_pll, best_eds = current, current_pll, current_eds
            best_score = current_score
            steps_since_imp = 0
        else:
            steps_since_imp += 1
        best_score_history.append(best_score)

        trajectory.append({
            "step": global_step, "phase": 2,
            "true_pll": current_pll, "best_pll": best_pll,
            "eds": current_eds, "best_eds": best_eds,
            "score": current_score, "best_score": best_score,
            "hamming_to_wt": hamming(current, wt_seq),
            "temperature": temp,
            "accepted": accepted, "accepted_downhill": accepted_downhill,
        })

        temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
        if steps_since_imp >= P2_PATIENCE: break
        if step >= P2_CONVERGENCE_WINDOW:
            if np.var(best_score_history[-P2_CONVERGENCE_WINDOW:]) < P2_VAR_THRESHOLD:
                break

    n_steps = len(trajectory) - 1
    final_ham = hamming(best_seq, wt_seq)
    print(f"    Phase 2 done: {n_steps} steps | "
          f"PLL: {trajectory[0]['true_pll']:.4f} -> {best_pll:.4f} | "
          f"Ham: {trajectory[0]['hamming_to_wt']} -> {final_ham}")

    return trajectory, best_seq, best_pll, best_eds


def run_control(start_seq, wt_seq, wt_emb, model, alphabet, bc, device, rng):
    current = start_seq
    current_pll = score_true_pll(current, model, alphabet, bc, device)
    cur_emb = get_embeddings(current, model, bc, device)
    current_eds = compute_eds(cur_emb, wt_emb)

    best_seq, best_pll, best_eds = current, current_pll, current_eds
    best_score_history = [best_pll]
    steps_since_imp = 0
    temp = CTRL_INITIAL_TEMP

    entropies = compute_entropy(current, model, alphabet, bc, device)

    trajectory = [{
        "step": 0, "phase": 0,
        "true_pll": current_pll, "best_pll": best_pll,
        "eds": current_eds, "best_eds": best_eds,
        "hamming_to_wt": hamming(current, wt_seq),
        "temperature": temp, "accepted": True, "accepted_downhill": False,
    }]

    for step in range(1, CTRL_MAX_STEPS + 1):
        props = propose_mutations(current, P2_PROPOSALS, rng, entropies)
        if not props:
            best_score_history.append(best_pll)
            steps_since_imp += 1
            trajectory.append({
                "step": step, "phase": 0,
                "true_pll": current_pll, "best_pll": best_pll,
                "eds": current_eds, "best_eds": best_eds,
                "hamming_to_wt": hamming(current, wt_seq),
                "temperature": temp, "accepted": False, "accepted_downhill": False,
            })
            temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
            if steps_since_imp >= P2_PATIENCE: break
            continue

        # MLLR pre-filter
        mllr = [(compute_mllr(current, p, a, model, alphabet, bc, device), p, a, s)
                for p, a, s in props]
        mllr.sort(key=lambda x: x[0], reverse=True)

        best_cand_pll = -float('inf'); best_cand_seq = None
        for m, pos, aa, seq in mllr[:P2_TOP_K]:
            cpll = score_true_pll(seq, model, alphabet, bc, device)
            if cpll > best_cand_pll:
                best_cand_pll = cpll; best_cand_seq = seq

        accepted = False; accepted_downhill = False
        if best_cand_seq is not None:
            if mcmc_accept(current_pll, best_cand_pll, temp, rng):
                accepted_downhill = (best_cand_pll < current_pll)
                current = best_cand_seq; current_pll = best_cand_pll
                accepted = True

        if current_pll > best_pll:
            best_seq, best_pll = current, current_pll
            steps_since_imp = 0
        else:
            steps_since_imp += 1
        best_score_history.append(best_pll)

        # EDS tracking (periodic)
        if step % 25 == 0:
            cur_emb = get_embeddings(current, model, bc, device)
            current_eds = compute_eds(cur_emb, wt_emb)
        if current_eds > best_eds:
            best_eds = current_eds

        trajectory.append({
            "step": step, "phase": 0,
            "true_pll": current_pll, "best_pll": best_pll,
            "eds": current_eds, "best_eds": best_eds,
            "hamming_to_wt": hamming(current, wt_seq),
            "temperature": temp,
            "accepted": accepted, "accepted_downhill": accepted_downhill,
        })

        temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
        if steps_since_imp >= P2_PATIENCE: break
        if step >= P2_CONVERGENCE_WINDOW:
            if np.var(best_score_history[-P2_CONVERGENCE_WINDOW:]) < P2_VAR_THRESHOLD:
                break

    n_steps = len(trajectory) - 1
    final_ham = hamming(best_seq, wt_seq)
    # Get final EDS for best
    bemb = get_embeddings(best_seq, model, bc, device)
    best_eds = compute_eds(bemb, wt_emb)
    return trajectory, best_seq, best_pll, best_eds


def main():
    t0 = time.perf_counter()
    rng = random.Random(SEED); np.random.seed(SEED)
    outdir = Path(OUTDIR); outdir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta_one(WT_FASTA); L = len(wt_seq)
    model, alphabet, bc = load_esm2(DEVICE)

    wt_pll = score_true_pll(wt_seq, model, alphabet, bc, DEVICE)
    wt_emb = get_embeddings(wt_seq, model, bc, DEVICE)

    print(f"WT length: {L} | WT PLL: {wt_pll:.4f}")
    print(f"Runs: {N_RUNS} two-phase + {N_RUNS} control = {N_RUNS*2} total")
    print(f"Phase 1: pure EDS, {P1_MAX_STEPS} steps, greedy")
    print(f"Phase 2: PLL+{P2_EDS_LAMBDA}*EDS, {P2_MAX_STEPS} steps, MCMC")
    print(f"Control: pure PLL, {CTRL_MAX_STEPS} steps, MCMC\n")

    start_seqs = [random_sequence(L, rng) for _ in range(N_RUNS)]

    all_results = []
    best_seqs = {}

    print("=" * 60)
    print("TWO-PHASE RUNS (Phase 1: EDS -> Phase 2: PLL+EDS)")
    print("=" * 60)

    for i in range(N_RUNS):
        print(f"\n── Two-phase run {i+1}/{N_RUNS} ──")
        start_seq = start_seqs[i]
        start_ham = hamming(start_seq, wt_seq)
        run_rng = random.Random(SEED + i)

        # Phase 1
        traj1, p1_best_seq, p1_best_eds = run_phase1(
            start_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE, run_rng)

        p1_steps = len(traj1) - 1
        p1_ham = hamming(p1_best_seq, wt_seq)

        # Phase 2: start from Phase 1's best
        traj2, p2_best_seq, p2_best_pll, p2_best_eds = run_phase2(
            p1_best_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE, run_rng,
            step_offset=p1_steps)

        # Merge trajectories
        full_traj = traj1 + traj2

        final_ham = hamming(p2_best_seq, wt_seq)
        bemb = get_embeddings(p2_best_seq, model, bc, DEVICE)
        final_eds = compute_eds(bemb, wt_emb)

        rid = len(all_results)
        all_results.append({
            "run_id": rid, "mode": "twophase",
            "start_hamming": start_ham,
            "p1_final_hamming": p1_ham, "p1_final_eds": p1_best_eds,
            "p1_steps": p1_steps,
            "final_pll": p2_best_pll, "final_hamming": final_ham,
            "final_eds": final_eds,
            "total_steps": len(full_traj) - 1,
            "trajectory": full_traj,
        })
        best_seqs[rid] = p2_best_seq

        print(f"  TOTAL: Ham {start_ham} -> {p1_ham} (P1) -> {final_ham} (P2) | "
              f"PLL={p2_best_pll:.4f} | EDS={final_eds:.1f}")

    print("\n" + "=" * 60)
    print("CONTROL RUNS (pure PLL from random)")
    print("=" * 60)

    for i in range(N_RUNS):
        print(f"\n── Control run {i+1}/{N_RUNS} ──")
        start_seq = start_seqs[i]  # same starting sequences
        start_ham = hamming(start_seq, wt_seq)
        run_rng = random.Random(SEED + 1000 + i)  # different rng stream

        traj, best_s, best_p, best_e = run_control(
            start_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE, run_rng)

        final_ham = hamming(best_s, wt_seq)

        rid = len(all_results)
        all_results.append({
            "run_id": rid, "mode": "control",
            "start_hamming": start_ham,
            "p1_final_hamming": None, "p1_final_eds": None,
            "p1_steps": 0,
            "final_pll": best_p, "final_hamming": final_ham,
            "final_eds": best_e,
            "total_steps": len(traj) - 1,
            "trajectory": traj,
        })
        best_seqs[rid] = best_s

        print(f"  Ham: {start_ham} -> {final_ham} | PLL={best_p:.4f} | EDS={best_e:.1f}")

    rows = []
    for r in all_results:
        for t in r["trajectory"]:
            row = dict(t); row["run_id"] = r["run_id"]; row["mode"] = r["mode"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "trajectory.csv", index=False)

    summary = [{k: v for k, v in r.items() if k != "trajectory"} for r in all_results]
    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    tp = [r for r in all_results if r["mode"] == "twophase"]
    ctrl = [r for r in all_results if r["mode"] == "control"]

    if tp:
        best_r = max(tp, key=lambda r: r["final_pll"])
        (outdir / "best_twophase.fasta").write_text(
            f">twophase_pll={best_r['final_pll']:.6f}_ham={best_r['final_hamming']}"
            f"_eds={best_r['final_eds']:.1f}\n"
            f"{best_seqs[best_r['run_id']]}\n", encoding="utf-8")
    if ctrl:
        best_r = max(ctrl, key=lambda r: r["final_pll"])
        (outdir / "best_control.fasta").write_text(
            f">control_pll={best_r['final_pll']:.6f}_ham={best_r['final_hamming']}"
            f"_eds={best_r['final_eds']:.1f}\n"
            f"{best_seqs[best_r['run_id']]}\n", encoding="utf-8")


    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Two-Phase Walk: Phase 1 (EDS) then Phase 2 (PLL+EDS)",
                 fontsize=13, fontweight="bold", y=1.02)

    # (0,0) EDS over time (shows Phase 1 improvement + Phase 2 maintenance)
    for r in tp:
        t = r["trajectory"]
        steps = [x["step"] for x in t]
        eds = [x.get("eds", x.get("best_eds", 0)) for x in t]
        p1_steps = [x["step"] for x in t if x["phase"] == 1]
        p1_eds = [x.get("eds", x.get("best_eds", 0)) for x in t if x["phase"] == 1]
        p2_steps = [x["step"] for x in t if x["phase"] == 2]
        p2_eds = [x.get("eds", x.get("best_eds", 0)) for x in t if x["phase"] == 2]
        axs[0, 0].plot(p1_steps, p1_eds, color=C_PHASE1, alpha=0.6, lw=1)
        if p2_steps:
            axs[0, 0].plot(p2_steps, p2_eds, color=C_PHASE2, alpha=0.6, lw=1)
    axs[0, 0].axhline(0, color="red", ls="--", label="WT=0")
    axs[0, 0].legend([Line2D([0],[0],color=C_PHASE1,lw=2),
                      Line2D([0],[0],color=C_PHASE2,lw=2),
                      Line2D([0],[0],color="red",ls="--")],
                     ["Phase 1 (EDS)", "Phase 2 (PLL+EDS)", "WT=0"], fontsize=8)
    axs[0, 0].set(title="EDS Over Time", xlabel="Step", ylabel="EDS")
    axs[0, 0].grid(True, alpha=0.3)

    # (0,1) Hamming over time
    for r in tp:
        t = r["trajectory"]
        p1_s = [x["step"] for x in t if x["phase"] == 1]
        p1_h = [x["hamming_to_wt"] for x in t if x["phase"] == 1]
        p2_s = [x["step"] for x in t if x["phase"] == 2]
        p2_h = [x["hamming_to_wt"] for x in t if x["phase"] == 2]
        axs[0, 1].plot(p1_s, p1_h, color=C_PHASE1, alpha=0.6, lw=1)
        if p2_s:
            axs[0, 1].plot(p2_s, p2_h, color=C_PHASE2, alpha=0.6, lw=1)
    axs[0, 1].set(title="Hamming to WT Over Time", xlabel="Step", ylabel="Hamming")
    axs[0, 1].grid(True, alpha=0.3)

    # (1,0) PLL over time (Phase 2 only — Phase 1 doesn't track PLL)
    for r in tp:
        t = r["trajectory"]
        p2 = [x for x in t if x["phase"] == 2]
        if p2:
            axs[1, 0].plot([x["step"] for x in p2], [x["best_pll"] for x in p2],
                           color=C_PHASE2, alpha=0.6, lw=1)
    axs[1, 0].axhline(wt_pll, color="red", ls="--", label=f"WT={wt_pll:.3f}")
    axs[1, 0].set(title="PLL During Phase 2", xlabel="Step", ylabel="Best PLL")
    axs[1, 0].legend(fontsize=9); axs[1, 0].grid(True, alpha=0.3)

    # (1,1) Phase 1 summary: start ham -> p1 ham -> final ham
    for r in tp:
        xs = [0, 1, 2]
        ys = [r["start_hamming"], r["p1_final_hamming"], r["final_hamming"]]
        axs[1, 1].plot(xs, ys, "o-", color=C_TWOPHASE, alpha=0.7, lw=2, markersize=8)
    axs[1, 1].set_xticks([0, 1, 2])
    axs[1, 1].set_xticklabels(["Random\nstart", "After\nPhase 1", "After\nPhase 2"])
    axs[1, 1].set(title="Hamming Distance Journey", ylabel="Hamming to WT")
    axs[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_phases.png", dpi=300, bbox_inches="tight"); plt.close()
    print(f"\nPlot 1 (phases):     {outdir / 'plots_phases.png'}")


    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Two-Phase vs Control (same starting sequences)",
                   fontsize=13, fontweight="bold", y=1.02)

    # (0,0) Final Hamming comparison
    tp_ham = [r["final_hamming"] for r in tp]
    ctrl_ham = [r["final_hamming"] for r in ctrl]
    bp = axs2[0, 0].boxplot([tp_ham, ctrl_ham], labels=["Two-phase", "Control"],
                             patch_artist=True)
    bp["boxes"][0].set_facecolor(C_TWOPHASE); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_CONTROL); bp["boxes"][1].set_alpha(0.7)
    axs2[0, 0].set(title="Final Hamming to WT (lower = closer)", ylabel="Hamming")
    axs2[0, 0].grid(True, alpha=0.3, axis="y")

    # (0,1) Final PLL comparison
    tp_pll = [r["final_pll"] for r in tp]
    ctrl_pll = [r["final_pll"] for r in ctrl]
    bp2 = axs2[0, 1].boxplot([tp_pll, ctrl_pll], labels=["Two-phase", "Control"],
                              patch_artist=True)
    bp2["boxes"][0].set_facecolor(C_TWOPHASE); bp2["boxes"][0].set_alpha(0.7)
    bp2["boxes"][1].set_facecolor(C_CONTROL); bp2["boxes"][1].set_alpha(0.7)
    axs2[0, 1].axhline(wt_pll, color="red", ls="--", label=f"WT={wt_pll:.3f}")
    axs2[0, 1].set(title="Final PLL (higher = better protein)", ylabel="PLL")
    axs2[0, 1].legend(fontsize=9); axs2[0, 1].grid(True, alpha=0.3, axis="y")

    # (1,0) Final EDS comparison
    tp_eds = [r["final_eds"] for r in tp]
    ctrl_eds = [r["final_eds"] for r in ctrl]
    bp3 = axs2[1, 0].boxplot([tp_eds, ctrl_eds], labels=["Two-phase", "Control"],
                              patch_artist=True)
    bp3["boxes"][0].set_facecolor(C_TWOPHASE); bp3["boxes"][0].set_alpha(0.7)
    bp3["boxes"][1].set_facecolor(C_CONTROL); bp3["boxes"][1].set_alpha(0.7)
    axs2[1, 0].axhline(0, color="red", ls="--", label="WT=0")
    axs2[1, 0].set(title="Final EDS (closer to 0 = closer to WT)", ylabel="EDS")
    axs2[1, 0].legend(fontsize=9); axs2[1, 0].grid(True, alpha=0.3, axis="y")

    # (1,1) Paired comparison: each run's two-phase vs control hamming
    for i in range(N_RUNS):
        axs2[1, 1].plot([0, 1], [tp[i]["final_hamming"], ctrl[i]["final_hamming"]],
                        "o-", color="gray", alpha=0.5, lw=1)
    axs2[1, 1].scatter([0]*N_RUNS, [r["final_hamming"] for r in tp],
                        c=C_TWOPHASE, s=80, edgecolors="black", zorder=3)
    axs2[1, 1].scatter([1]*N_RUNS, [r["final_hamming"] for r in ctrl],
                        c=C_CONTROL, s=80, edgecolors="black", zorder=3)
    axs2[1, 1].set_xticks([0, 1]); axs2[1, 1].set_xticklabels(["Two-phase", "Control"])
    axs2[1, 1].set(title="Paired Hamming (same start seq)", ylabel="Final Hamming")
    axs2[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_comparison.png", dpi=300, bbox_inches="tight"); plt.close()
    print(f"Plot 2 (comparison): {outdir / 'plots_comparison.png'}")

    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("EDS Analysis", fontsize=13, fontweight="bold")

    # (0) EDS trajectories — all runs
    for r in tp:
        t = r["trajectory"]
        axs3[0].plot([x["step"] for x in t],
                     [x.get("eds", x.get("best_eds", 0)) for x in t],
                     color=C_TWOPHASE, alpha=0.5, lw=0.8)
    for r in ctrl:
        t = r["trajectory"]
        axs3[0].plot([x["step"] for x in t], [x["eds"] for x in t],
                     color=C_CONTROL, alpha=0.5, lw=0.8)
    axs3[0].axhline(0, color="red", ls="--", label="WT=0")
    axs3[0].legend([Line2D([0],[0],color=C_TWOPHASE,lw=2),
                    Line2D([0],[0],color=C_CONTROL,lw=2),
                    Line2D([0],[0],color="red",ls="--")],
                   ["Two-phase", "Control", "WT=0"], fontsize=8)
    axs3[0].set(title="EDS Trajectories", xlabel="Step", ylabel="EDS")
    axs3[0].grid(True, alpha=0.3)

    # (1) Final EDS vs PLL scatter
    for r in tp:
        axs3[1].scatter(r["final_eds"], r["final_pll"],
                        c=C_TWOPHASE, s=80, edgecolors="black", lw=0.5)
    for r in ctrl:
        axs3[1].scatter(r["final_eds"], r["final_pll"],
                        c=C_CONTROL, s=80, edgecolors="black", lw=0.5)
    axs3[1].legend([Line2D([0],[0],marker="o",color="w",markerfacecolor=C_TWOPHASE,markersize=8),
                    Line2D([0],[0],marker="o",color="w",markerfacecolor=C_CONTROL,markersize=8)],
                   ["Two-phase", "Control"], fontsize=8)
    axs3[1].set(title="Final EDS vs PLL", xlabel="EDS", ylabel="PLL")
    axs3[1].grid(True, alpha=0.3)

    # (2) Phase 1 EDS improvement
    p1_start_eds = [r["trajectory"][0].get("eds", 0) for r in tp]
    p1_end_eds = [r["p1_final_eds"] for r in tp]
    for i in range(N_RUNS):
        axs3[2].plot([0, 1], [p1_start_eds[i], p1_end_eds[i]],
                     "o-", color=C_PHASE1, alpha=0.7, lw=2, markersize=8)
    axs3[2].axhline(0, color="red", ls="--", alpha=0.5)
    axs3[2].set_xticks([0, 1]); axs3[2].set_xticklabels(["Start", "After Phase 1"])
    axs3[2].set(title="Phase 1: EDS Improvement", ylabel="EDS")
    axs3[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_eds.png", dpi=300, bbox_inches="tight"); plt.close()
    print(f"Plot 3 (EDS):        {outdir / 'plots_eds.png'}")

    print(f"\n{'='*65}")
    print(f"WT PLL: {wt_pll:.4f} | WT EDS: 0.00\n")

    print("TWO-PHASE (EDS approach -> PLL+EDS refine):")
    print(f"  Hamming: start={np.mean([r['start_hamming'] for r in tp]):.0f}"
          f" -> P1={np.mean([r['p1_final_hamming'] for r in tp]):.0f}"
          f" -> final={np.mean([r['final_hamming'] for r in tp]):.0f}")
    print(f"  PLL:  mean={np.mean(tp_pll):.4f}  best={max(tp_pll):.4f}")
    print(f"  EDS:  mean={np.mean(tp_eds):.1f}  best={max(tp_eds):.1f}")

    print(f"\nCONTROL (pure PLL):")
    print(f"  Hamming: start={np.mean([r['start_hamming'] for r in ctrl]):.0f}"
          f" -> final={np.mean([r['final_hamming'] for r in ctrl]):.0f}")
    print(f"  PLL:  mean={np.mean(ctrl_pll):.4f}  best={max(ctrl_pll):.4f}")
    print(f"  EDS:  mean={np.mean(ctrl_eds):.1f}  best={max(ctrl_eds):.1f}")

    print(f"\nDIFFERENCE (two-phase minus control):")
    print(f"  Hamming: {np.mean(tp_ham) - np.mean(ctrl_ham):+.1f}")
    print(f"  PLL:     {np.mean(tp_pll) - np.mean(ctrl_pll):+.4f}")
    print(f"  EDS:     {np.mean(tp_eds) - np.mean(ctrl_eds):+.1f}")

    print(f"\nRuntime: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"Outputs: {outdir}/")


if __name__ == "__main__":
    main()