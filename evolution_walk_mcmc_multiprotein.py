"""
Evolution walk v5: Multi-protein two-phase analysis.

Loops over a list of target proteins. For each protein:
  - Phase 1: pure EDS greedy approach from random starts
  - Phase 2: PLL+EDS MCMC refinement
  - Control: pure PLL MCMC from same random starts

Each protein gets its own output directory with plots and CSVs.
A cross-protein comparison summary is saved at the top level.
"""
import os
import warnings
from pathlib import Path
import random
import math
import time
import sys

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

FASTA_FILES = [
    ("insulin",     "insulin_fastas/insulin_human_wt.fasta"),
    # Add more proteins here:
    # ("lysozyme",    "fastas/lysozyme.fasta"),
    # ("ubiquitin",   "fastas/ubiquitin.fasta"),
    # ("igG_heavy",   "fastas/igg_heavy.fasta"),
    # ("GFP",         "fastas/gfp.fasta"),
]

OUTDIR_ROOT = "evolution_walk_v5_multiprotein"

REPR_LAYER = 33
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

N_RUNS = 5

# Phase 1: Pure EDS approach
P1_MAX_STEPS = 500
P1_PROPOSALS = 30
P1_PATIENCE = 100
P1_HAMMING_TARGET = 20

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

# Control
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

    trajectory = [{"step": 0, "phase": 1, "eds": current_eds, "best_eds": best_eds,
                   "hamming_to_wt": hamming(current, wt_seq), "accepted": True}]

    for step in range(1, P1_MAX_STEPS + 1):
        props = propose_mutations(current, P1_PROPOSALS, rng)
        if not props:
            steps_since_imp += 1
            trajectory.append({"step": step, "phase": 1, "eds": current_eds,
                               "best_eds": best_eds,
                               "hamming_to_wt": hamming(current, wt_seq),
                               "accepted": False})
            if steps_since_imp >= P1_PATIENCE: break
            continue

        best_cand_eds = -float('inf'); best_cand_seq = None
        for pos, aa, seq in props:
            cemb = get_embeddings(seq, model, bc, device)
            ceds = compute_eds(cemb, wt_emb)
            if ceds > best_cand_eds:
                best_cand_eds = ceds; best_cand_seq = seq

        accepted = False
        if best_cand_seq is not None and best_cand_eds > current_eds:
            current = best_cand_seq; current_eds = best_cand_eds; accepted = True

        if current_eds > best_eds:
            best_seq, best_eds = current, current_eds; steps_since_imp = 0
        else:
            steps_since_imp += 1

        cur_ham = hamming(current, wt_seq)
        trajectory.append({"step": step, "phase": 1, "eds": current_eds,
                           "best_eds": best_eds, "hamming_to_wt": cur_ham,
                           "accepted": accepted})

        if steps_since_imp >= P1_PATIENCE: break
        if cur_ham <= P1_HAMMING_TARGET: break

    return trajectory, best_seq, best_eds


def run_phase2(start_seq, wt_seq, wt_emb, model, alphabet, bc, device, rng,
               step_offset=0):
    current = start_seq
    current_pll = score_true_pll(current, model, alphabet, bc, device)
    cur_emb = get_embeddings(current, model, bc, device)
    current_eds = compute_eds(cur_emb, wt_emb)
    current_score = current_pll + P2_EDS_LAMBDA * current_eds

    best_seq, best_pll, best_eds = current, current_pll, current_eds
    best_score = current_score
    best_score_history = [best_score]
    steps_since_imp = 0
    temp = P2_INITIAL_TEMP

    entropies = compute_entropy(current, model, alphabet, bc, device)

    trajectory = [{"step": step_offset, "phase": 2, "true_pll": current_pll,
                   "best_pll": best_pll, "eds": current_eds, "best_eds": best_eds,
                   "hamming_to_wt": hamming(current, wt_seq), "temperature": temp,
                   "accepted": True, "accepted_downhill": False}]

    for step in range(1, P2_MAX_STEPS + 1):
        gs = step_offset + step
        props = propose_mutations(current, P2_PROPOSALS, rng, entropies)
        if not props:
            best_score_history.append(best_score); steps_since_imp += 1
            trajectory.append({"step": gs, "phase": 2, "true_pll": current_pll,
                               "best_pll": best_pll, "eds": current_eds,
                               "best_eds": best_eds,
                               "hamming_to_wt": hamming(current, wt_seq),
                               "temperature": temp, "accepted": False,
                               "accepted_downhill": False})
            temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
            if steps_since_imp >= P2_PATIENCE: break
            continue

        mllr = [(compute_mllr(current, p, a, model, alphabet, bc, device), p, a, s)
                for p, a, s in props]
        mllr.sort(key=lambda x: x[0], reverse=True)

        best_c_score = -float('inf'); best_c_seq = None
        best_c_pll = -float('inf'); best_c_eds = current_eds
        for m, pos, aa, seq in mllr[:P2_TOP_K]:
            cpll = score_true_pll(seq, model, alphabet, bc, device)
            cemb = get_embeddings(seq, model, bc, device)
            ceds = compute_eds(cemb, wt_emb)
            cs = cpll + P2_EDS_LAMBDA * ceds
            if cs > best_c_score:
                best_c_score = cs; best_c_seq = seq
                best_c_pll = cpll; best_c_eds = ceds

        accepted = False; accepted_downhill = False
        if best_c_seq is not None:
            if mcmc_accept(current_score, best_c_score, temp, rng):
                accepted_downhill = (best_c_score < current_score)
                current = best_c_seq; current_pll = best_c_pll
                current_eds = best_c_eds; current_score = best_c_score
                accepted = True

        if current_score > best_score:
            best_seq, best_pll, best_eds = current, current_pll, current_eds
            best_score = current_score; steps_since_imp = 0
        else:
            steps_since_imp += 1
        best_score_history.append(best_score)

        trajectory.append({"step": gs, "phase": 2, "true_pll": current_pll,
                           "best_pll": best_pll, "eds": current_eds,
                           "best_eds": best_eds,
                           "hamming_to_wt": hamming(current, wt_seq),
                           "temperature": temp, "accepted": accepted,
                           "accepted_downhill": accepted_downhill})

        temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
        if steps_since_imp >= P2_PATIENCE: break
        if step >= P2_CONVERGENCE_WINDOW:
            if np.var(best_score_history[-P2_CONVERGENCE_WINDOW:]) < P2_VAR_THRESHOLD:
                break

    return trajectory, best_seq, best_pll, best_eds


def run_control(start_seq, wt_seq, wt_emb, model, alphabet, bc, device, rng):
    current = start_seq
    current_pll = score_true_pll(current, model, alphabet, bc, device)
    cur_emb = get_embeddings(current, model, bc, device)
    current_eds = compute_eds(cur_emb, wt_emb)

    best_seq, best_pll, best_eds = current, current_pll, current_eds
    best_score_history = [best_pll]
    steps_since_imp = 0; temp = CTRL_INITIAL_TEMP

    entropies = compute_entropy(current, model, alphabet, bc, device)

    trajectory = [{"step": 0, "phase": 0, "true_pll": current_pll,
                   "best_pll": best_pll, "eds": current_eds, "best_eds": best_eds,
                   "hamming_to_wt": hamming(current, wt_seq), "temperature": temp,
                   "accepted": True, "accepted_downhill": False}]

    for step in range(1, CTRL_MAX_STEPS + 1):
        props = propose_mutations(current, P2_PROPOSALS, rng, entropies)
        if not props:
            best_score_history.append(best_pll); steps_since_imp += 1
            trajectory.append({"step": step, "phase": 0, "true_pll": current_pll,
                               "best_pll": best_pll, "eds": current_eds,
                               "best_eds": best_eds,
                               "hamming_to_wt": hamming(current, wt_seq),
                               "temperature": temp, "accepted": False,
                               "accepted_downhill": False})
            temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
            if steps_since_imp >= P2_PATIENCE: break
            continue

        mllr = [(compute_mllr(current, p, a, model, alphabet, bc, device), p, a, s)
                for p, a, s in props]
        mllr.sort(key=lambda x: x[0], reverse=True)

        best_c_pll = -float('inf'); best_c_seq = None
        for m, pos, aa, seq in mllr[:P2_TOP_K]:
            cpll = score_true_pll(seq, model, alphabet, bc, device)
            if cpll > best_c_pll: best_c_pll = cpll; best_c_seq = seq

        accepted = False; accepted_downhill = False
        if best_c_seq is not None:
            if mcmc_accept(current_pll, best_c_pll, temp, rng):
                accepted_downhill = (best_c_pll < current_pll)
                current = best_c_seq; current_pll = best_c_pll; accepted = True

        if current_pll > best_pll:
            best_seq, best_pll = current, current_pll; steps_since_imp = 0
        else:
            steps_since_imp += 1
        best_score_history.append(best_pll)

        if step % 25 == 0:
            cur_emb = get_embeddings(current, model, bc, device)
            current_eds = compute_eds(cur_emb, wt_emb)
        if current_eds > best_eds: best_eds = current_eds

        trajectory.append({"step": step, "phase": 0, "true_pll": current_pll,
                           "best_pll": best_pll, "eds": current_eds,
                           "best_eds": best_eds,
                           "hamming_to_wt": hamming(current, wt_seq),
                           "temperature": temp, "accepted": accepted,
                           "accepted_downhill": accepted_downhill})

        temp = max(temp * P2_COOLING_RATE, P2_MIN_TEMP)
        if steps_since_imp >= P2_PATIENCE: break
        if step >= P2_CONVERGENCE_WINDOW:
            if np.var(best_score_history[-P2_CONVERGENCE_WINDOW:]) < P2_VAR_THRESHOLD:
                break

    bemb = get_embeddings(best_seq, model, bc, device)
    best_eds = compute_eds(bemb, wt_emb)
    return trajectory, best_seq, best_pll, best_eds


def run_protein(name, fasta_path, model, alphabet, bc, root_outdir, master_rng):
    """Run full two-phase + control experiment for one protein."""
    t0 = time.perf_counter()
    outdir = root_outdir / name
    outdir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta_one(fasta_path); L = len(wt_seq)
    wt_pll = score_true_pll(wt_seq, model, alphabet, bc, DEVICE)
    wt_emb = get_embeddings(wt_seq, model, bc, DEVICE)

    print(f"\n{'#'*65}")
    print(f"# PROTEIN: {name} | Length: {L} | WT PLL: {wt_pll:.4f}")
    print(f"{'#'*65}")

    start_seqs = [random_sequence(L, master_rng) for _ in range(N_RUNS)]
    all_results = []; best_seqs = {}

    # Two-phase runs
    print(f"\n  Two-phase runs:")
    for i in range(N_RUNS):
        start_seq = start_seqs[i]
        start_ham = hamming(start_seq, wt_seq)
        run_rng = random.Random(SEED + i)

        traj1, p1_seq, p1_eds = run_phase1(
            start_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE, run_rng)
        p1_steps = len(traj1) - 1
        p1_ham = hamming(p1_seq, wt_seq)

        traj2, p2_seq, p2_pll, p2_eds = run_phase2(
            p1_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE, run_rng,
            step_offset=p1_steps)

        final_ham = hamming(p2_seq, wt_seq)
        bemb = get_embeddings(p2_seq, model, bc, DEVICE)
        final_eds = compute_eds(bemb, wt_emb)

        rid = len(all_results)
        all_results.append({
            "run_id": rid, "mode": "twophase", "protein": name,
            "length": L, "wt_pll": wt_pll,
            "start_hamming": start_ham,
            "p1_final_hamming": p1_ham, "p1_final_eds": p1_eds, "p1_steps": p1_steps,
            "final_pll": p2_pll, "final_hamming": final_ham, "final_eds": final_eds,
            "total_steps": len(traj1) + len(traj2) - 2,
            "trajectory": traj1 + traj2,
        })
        best_seqs[rid] = p2_seq
        print(f"    Run {i+1}/{N_RUNS}: Ham {start_ham}->{p1_ham}->{final_ham} | "
              f"PLL={p2_pll:.4f} | EDS={final_eds:.1f}")

    # Control runs
    print(f"  Control runs:")
    for i in range(N_RUNS):
        start_seq = start_seqs[i]
        start_ham = hamming(start_seq, wt_seq)
        run_rng = random.Random(SEED + 1000 + i)

        traj, best_s, best_p, best_e = run_control(
            start_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE, run_rng)
        final_ham = hamming(best_s, wt_seq)

        rid = len(all_results)
        all_results.append({
            "run_id": rid, "mode": "control", "protein": name,
            "length": L, "wt_pll": wt_pll,
            "start_hamming": start_ham,
            "p1_final_hamming": None, "p1_final_eds": None, "p1_steps": 0,
            "final_pll": best_p, "final_hamming": final_ham, "final_eds": best_e,
            "total_steps": len(traj) - 1,
            "trajectory": traj,
        })
        best_seqs[rid] = best_s
        print(f"    Run {i+1}/{N_RUNS}: Ham {start_ham}->{final_ham} | "
              f"PLL={best_p:.4f} | EDS={best_e:.1f}")

    rows = []
    for r in all_results:
        for t in r["trajectory"]:
            row = dict(t); row["run_id"] = r["run_id"]; row["mode"] = r["mode"]
            row["protein"] = name
            rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "trajectory.csv", index=False)

    summary = [{k: v for k, v in r.items() if k != "trajectory"} for r in all_results]
    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    # ── Save FASTAs ──
    tp = [r for r in all_results if r["mode"] == "twophase"]
    ctrl = [r for r in all_results if r["mode"] == "control"]

    if tp:
        best_r = max(tp, key=lambda r: r["final_pll"])
        (outdir / "best_twophase.fasta").write_text(
            f">{name}_twophase_pll={best_r['final_pll']:.6f}_ham={best_r['final_hamming']}\n"
            f"{best_seqs[best_r['run_id']]}\n", encoding="utf-8")
    if ctrl:
        best_r = max(ctrl, key=lambda r: r["final_pll"])
        (outdir / "best_control.fasta").write_text(
            f">{name}_control_pll={best_r['final_pll']:.6f}_ham={best_r['final_hamming']}\n"
            f"{best_seqs[best_r['run_id']]}\n", encoding="utf-8")


    # Plot 1: Phase transition (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} (L={L}): Two-Phase Walk", fontsize=13, fontweight="bold", y=1.02)

    for r in tp:
        t = r["trajectory"]
        p1 = [x for x in t if x["phase"] == 1]
        p2 = [x for x in t if x["phase"] == 2]
        axs[0,0].plot([x["step"] for x in p1], [x["eds"] for x in p1],
                      color=C_PHASE1, alpha=0.6, lw=1)
        if p2: axs[0,0].plot([x["step"] for x in p2],
                             [x["eds"] for x in p2], color=C_PHASE2, alpha=0.6, lw=1)
    axs[0,0].axhline(0, color="red", ls="--"); axs[0,0].set(title="EDS", xlabel="Step")
    axs[0,0].grid(True, alpha=0.3)

    for r in tp:
        t = r["trajectory"]
        p1 = [x for x in t if x["phase"] == 1]
        p2 = [x for x in t if x["phase"] == 2]
        axs[0,1].plot([x["step"] for x in p1], [x["hamming_to_wt"] for x in p1],
                      color=C_PHASE1, alpha=0.6, lw=1)
        if p2: axs[0,1].plot([x["step"] for x in p2],
                             [x["hamming_to_wt"] for x in p2], color=C_PHASE2, alpha=0.6, lw=1)
    axs[0,1].set(title="Hamming to WT", xlabel="Step"); axs[0,1].grid(True, alpha=0.3)

    for r in tp:
        p2 = [x for x in r["trajectory"] if x["phase"] == 2]
        if p2: axs[1,0].plot([x["step"] for x in p2], [x["best_pll"] for x in p2],
                             color=C_PHASE2, alpha=0.6, lw=1)
    axs[1,0].axhline(wt_pll, color="red", ls="--", label=f"WT={wt_pll:.3f}")
    axs[1,0].set(title="PLL (Phase 2)", xlabel="Step"); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)

    for r in tp:
        axs[1,1].plot([0, 1, 2],
                      [r["start_hamming"], r["p1_final_hamming"], r["final_hamming"]],
                      "o-", color=C_TWOPHASE, alpha=0.7, lw=2, markersize=8)
    axs[1,1].set_xticks([0,1,2]); axs[1,1].set_xticklabels(["Start", "P1", "P2"])
    axs[1,1].set(title="Hamming Journey", ylabel="Hamming"); axs[1,1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_phases.png", dpi=300, bbox_inches="tight"); plt.close()

    # Plot 2: Comparison (2x2)
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f"{name}: Two-Phase vs Control", fontsize=13, fontweight="bold", y=1.02)

    tp_h = [r["final_hamming"] for r in tp]; ct_h = [r["final_hamming"] for r in ctrl]
    tp_p = [r["final_pll"] for r in tp]; ct_p = [r["final_pll"] for r in ctrl]
    tp_e = [r["final_eds"] for r in tp]; ct_e = [r["final_eds"] for r in ctrl]

    for idx, (data, title, yl) in enumerate([
        ([tp_h, ct_h], "Final Hamming (lower=closer)", "Hamming"),
        ([tp_p, ct_p], "Final PLL (higher=better)", "PLL"),
        ([tp_e, ct_e], "Final EDS (closer to 0=closer)", "EDS"),
    ]):
        ax = axs2[idx // 2, idx % 2]
        bp = ax.boxplot(data, tick_labels=["Two-phase", "Control"], patch_artist=True)
        bp["boxes"][0].set_facecolor(C_TWOPHASE); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(C_CONTROL); bp["boxes"][1].set_alpha(0.7)
        ax.set(title=title, ylabel=yl); ax.grid(True, alpha=0.3, axis="y")
        if "PLL" in title:
            ax.axhline(wt_pll, color="red", ls="--", label=f"WT={wt_pll:.3f}"); ax.legend()
        if "EDS" in title:
            ax.axhline(0, color="red", ls="--", label="WT=0"); ax.legend()

    for i in range(N_RUNS):
        axs2[1,1].plot([0, 1], [tp[i]["final_hamming"], ctrl[i]["final_hamming"]],
                       "o-", color="gray", alpha=0.5, lw=1)
    axs2[1,1].scatter([0]*N_RUNS, tp_h, c=C_TWOPHASE, s=80, edgecolors="black", zorder=3)
    axs2[1,1].scatter([1]*N_RUNS, ct_h, c=C_CONTROL, s=80, edgecolors="black", zorder=3)
    axs2[1,1].set_xticks([0,1]); axs2[1,1].set_xticklabels(["Two-phase", "Control"])
    axs2[1,1].set(title="Paired Hamming", ylabel="Hamming"); axs2[1,1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_comparison.png", dpi=300, bbox_inches="tight"); plt.close()

    runtime = (time.perf_counter() - t0) / 60

    return {
        "protein": name, "length": L, "wt_pll": wt_pll,
        "tp_mean_pll": np.mean(tp_p), "tp_mean_ham": np.mean(tp_h),
        "tp_mean_eds": np.mean(tp_e),
        "tp_mean_p1_ham": np.mean([r["p1_final_hamming"] for r in tp]),
        "tp_ham_reduction": np.mean([r["start_hamming"] - r["final_hamming"] for r in tp]),
        "ctrl_mean_pll": np.mean(ct_p), "ctrl_mean_ham": np.mean(ct_h),
        "ctrl_mean_eds": np.mean(ct_e),
        "ctrl_ham_reduction": np.mean([r["start_hamming"] - r["final_hamming"] for r in ctrl]),
        "ham_advantage": np.mean(ct_h) - np.mean(tp_h),
        "eds_advantage": np.mean(tp_e) - np.mean(ct_e),
        "runtime_min": runtime,
    }


def main():
    t0 = time.perf_counter()
    master_rng = random.Random(SEED); np.random.seed(SEED)
    root = Path(OUTDIR_ROOT); root.mkdir(parents=True, exist_ok=True)

    # Validate all FASTA files exist before loading model
    for name, path in FASTA_FILES:
        if not Path(path).exists():
            print(f"ERROR: FASTA file not found: {path}")
            sys.exit(1)

    print(f"Proteins to process: {[n for n, _ in FASTA_FILES]}")
    print(f"Runs per protein: {N_RUNS} two-phase + {N_RUNS} control")
    print(f"Loading ESM2 model...")
    model, alphabet, bc = load_esm2(DEVICE)
    print(f"Model loaded on {DEVICE}\n")

    cross_protein = []

    for name, fasta_path in FASTA_FILES:
        result = run_protein(name, fasta_path, model, alphabet, bc, root, master_rng)
        cross_protein.append(result)
        print(f"\n  {name} done in {result['runtime_min']:.1f} min")

    df_cross = pd.DataFrame(cross_protein)
    df_cross.to_csv(root / "cross_protein_summary.csv", index=False)

    if len(cross_protein) > 1:
        # Cross-protein comparison plot
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Cross-Protein Comparison", fontsize=13, fontweight="bold", y=1.02)

        names = [r["protein"] for r in cross_protein]
        x = np.arange(len(names))
        w = 0.35

        # (0,0) Hamming reduction: two-phase vs control
        tp_red = [r["tp_ham_reduction"] for r in cross_protein]
        ct_red = [r["ctrl_ham_reduction"] for r in cross_protein]
        axs[0,0].bar(x - w/2, tp_red, w, color=C_TWOPHASE, label="Two-phase", alpha=0.8)
        axs[0,0].bar(x + w/2, ct_red, w, color=C_CONTROL, label="Control", alpha=0.8)
        axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(names, rotation=30, ha="right")
        axs[0,0].set(title="Hamming Reduction (+ve = toward WT)", ylabel="Reduction")
        axs[0,0].legend(); axs[0,0].axhline(0, color="black", lw=0.5)
        axs[0,0].grid(True, alpha=0.3, axis="y")

        # (0,1) Two-phase advantage in Hamming
        adv = [r["ham_advantage"] for r in cross_protein]
        colors = [C_TWOPHASE if a > 0 else C_CONTROL for a in adv]
        axs[0,1].bar(x, adv, color=colors, alpha=0.8, edgecolor="black")
        axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(names, rotation=30, ha="right")
        axs[0,1].set(title="Two-Phase Hamming Advantage", ylabel="Control Ham - TwoPhase Ham")
        axs[0,1].axhline(0, color="black", lw=0.5)
        axs[0,1].grid(True, alpha=0.3, axis="y")

        # (1,0) EDS advantage
        eds_adv = [r["eds_advantage"] for r in cross_protein]
        colors_e = [C_TWOPHASE if a > 0 else C_CONTROL for a in eds_adv]
        axs[1,0].bar(x, eds_adv, color=colors_e, alpha=0.8, edgecolor="black")
        axs[1,0].set_xticks(x); axs[1,0].set_xticklabels(names, rotation=30, ha="right")
        axs[1,0].set(title="Two-Phase EDS Advantage (+ve = closer to WT)", ylabel="EDS diff")
        axs[1,0].axhline(0, color="black", lw=0.5)
        axs[1,0].grid(True, alpha=0.3, axis="y")

        # (1,1) Length vs Hamming reduction scatter
        lengths = [r["length"] for r in cross_protein]
        axs[1,1].scatter(lengths, tp_red, c=C_TWOPHASE, s=100, edgecolors="black",
                         label="Two-phase", zorder=3)
        axs[1,1].scatter(lengths, ct_red, c=C_CONTROL, s=100, edgecolors="black",
                         label="Control", zorder=3)
        for i, n in enumerate(names):
            axs[1,1].annotate(n, (lengths[i], tp_red[i]), fontsize=8,
                              xytext=(5, 5), textcoords="offset points")
        axs[1,1].set(title="Protein Length vs Hamming Reduction",
                     xlabel="Protein Length", ylabel="Hamming Reduction")
        axs[1,1].legend(); axs[1,1].axhline(0, color="black", lw=0.5)
        axs[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(root / "plots_cross_protein.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nCross-protein plot: {root / 'plots_cross_protein.png'}")


    print(f"\n{'='*70}")
    print(f"CROSS-PROTEIN SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Protein':<15} {'Len':>4} {'WT PLL':>8} "
          f"{'TP Ham':>7} {'CT Ham':>7} {'Adv':>5} "
          f"{'TP EDS':>7} {'CT EDS':>7} {'Time':>6}")
    print("-" * 75)
    for r in cross_protein:
        print(f"{r['protein']:<15} {r['length']:>4} {r['wt_pll']:>+8.3f} "
              f"{r['tp_mean_ham']:>7.0f} {r['ctrl_mean_ham']:>7.0f} "
              f"{r['ham_advantage']:>+5.1f} "
              f"{r['tp_mean_eds']:>7.1f} {r['ctrl_mean_eds']:>7.1f} "
              f"{r['runtime_min']:>5.0f}m")

    total_time = (time.perf_counter() - t0) / 60
    print(f"\nTotal runtime: {total_time:.1f} min")
    print(f"Outputs: {root}/")


if __name__ == "__main__":
    main()