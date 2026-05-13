"""
Evolution walk v4 — Multi-protein


The identity at position j influences
ESM2's prediction at position i. The coupling measurement is
double-masking:

    coupling(i, j) = JSD( p(aa_i | seq_{-i}),  p(aa_i | seq_{-i,-j}) )

If masking j changes the prediction at i, they're coupled. 
"""

import os
import warnings
import sys
from pathlib import Path
import random
import math
import time
import json
from typing import List, Tuple, Dict, Optional
from itertools import combinations

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*libomp.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import esm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import jensenshannon


FASTA_DIR = "target_fastas"           
OUTDIR = "evolution_walk_v10b_multi_v3"

ESM2_MODEL = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

RUNS_PER_PROTEIN = 2



INITIAL_TEMP = 0.3
COOLING_RATE = 0.997               


LAMBDA_MIN = 0.1                    
LAMBDA_CAP = 0.5                    
LAMBDA_COUPLING_SCALE = 5.0        
LAMBDA_RAMP_FRAC = 0.4               
                                

PROPOSALS_PER_STEP = 40               

COUPLING_RECOMPUTE_EVERY = 150        

COUPLING_THRESHOLD = 0.05            

COUPLING_MIN_FRAC = 0.20             


def get_coupling_sample_pairs(L):
    return min(2000, max(500, L * 3))


PLL_DIAGNOSTIC_EVERY = 500         
LOG_EVERY = 50                      

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")



def get_protein_config(L: int) -> dict:
    cfg = {}
 
    cfg["k_single"] = max(1, round(L / 100))
    
    cfg["group_size_min"] = max(3, round(L / 80))
    cfg["group_size_max"] = max(5, round(L / 40))

    avg_group = (cfg["group_size_min"] + cfg["group_size_max"]) / 2
    cfg["n_groups"] = max(15, int(0.7 * L / avg_group))

    cfg["p_coupled"] = min(0.8, 0.3 + L / 1000)

    cfg["max_steps"] = max(2000, 10 * L)
    
    cfg["cooling_rate"] = (0.005 / INITIAL_TEMP) ** (1.0 / cfg["max_steps"])
    
    cfg["prcs_window"] = max(20, min(50, L // 4))
    
    return cfg


def read_fasta_one(path: str) -> Tuple[str, str]:
    lines = Path(path).read_text().strip().splitlines()
    name = lines[0].lstrip(">").strip().split()[0]
    seq = "".join(ln.strip() for ln in lines[1:] if not ln.startswith(">"))
    seq = seq.replace(" ", "").upper()
    return name, seq


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def mutate_positions(seq: str, positions: List[int], aas: List[str]) -> str:
    s = list(seq)
    for pos, aa in zip(positions, aas):
        s[pos] = aa
    return "".join(s)


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def load_esm2(device: str):
    print("Loading ESM2...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(torch.device(device))
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded on {device}")
    return model, alphabet, batch_converter



@torch.no_grad()
def get_per_residue_embedding(seq, model, alphabet, batch_converter, device):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    return out["representations"][REPR_LAYER][0, 1:-1, :]  # (L, 1280)


def score_prcs(emb_candidate, emb_wt):
    per_pos = F.cosine_similarity(emb_candidate, emb_wt, dim=1)  # (L,)
    return per_pos.mean().item(), per_pos.cpu().numpy()


def score_epistatic(emb_candidate, emb_wt, groups):
    if not groups:
        return 0.0
    
    all_cosines = []
    for group in groups:
        for i, j in combinations(group, 2):
            diff_cand = emb_candidate[i] - emb_candidate[j]
            diff_wt = emb_wt[i] - emb_wt[j]
            cos = F.cosine_similarity(diff_cand.unsqueeze(0),
                                       diff_wt.unsqueeze(0)).item()
            all_cosines.append(cos)
    
    return np.mean(all_cosines) if all_cosines else 0.0


def compute_combined_score(emb_candidate, emb_wt, groups, lam):
    prcs, per_pos = score_prcs(emb_candidate, emb_wt)
    epi = score_epistatic(emb_candidate, emb_wt, groups) if lam > 0.01 else 0.0
    combined = prcs + lam * epi
    return combined, prcs, epi, per_pos


@torch.no_grad()
def score_true_pll(seq, model, alphabet, batch_converter, device,
                   pos_batch_size=16):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    L = len(seq)
    
    log_probs = []
    for start in range(0, L, pos_batch_size):
        end = min(start + pos_batch_size, L)
        batch_tokens = tokens.repeat(end - start, 1)
        for idx, pos in enumerate(range(start, end)):
            batch_tokens[idx, pos + 1] = alphabet.mask_idx
        
        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = out["logits"]
        
        for idx, pos in enumerate(range(start, end)):
            log_p = torch.log_softmax(logits[idx, pos + 1], dim=-1)
            true_aa = tokens[0, pos + 1].item()
            log_probs.append(log_p[true_aa].item())
    
    return np.mean(log_probs)


@torch.no_grad()
def compute_single_masked_marginals(seq, model, alphabet, batch_converter,
                                     device, positions):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    marginals = {}
    
    for pos in positions:
        masked = tokens.clone()
        masked[0, pos + 1] = alphabet.mask_idx
        out = model(masked, repr_layers=[], return_contacts=False)
        logits = out["logits"][0, pos + 1]
        probs = torch.softmax(logits, dim=-1)
        marginals[pos] = probs[aa_indices].cpu().numpy()
    
    return marginals


@torch.no_grad()
def compute_double_masked_marginals(seq, model, alphabet, batch_converter,
                                     device, pos_i, pos_j):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    
    masked = tokens.clone()
    masked[0, pos_i + 1] = alphabet.mask_idx
    masked[0, pos_j + 1] = alphabet.mask_idx
    out = model(masked, repr_layers=[], return_contacts=False)
    
    logits_i = out["logits"][0, pos_i + 1]
    probs_i = torch.softmax(logits_i, dim=-1)
    
    logits_j = out["logits"][0, pos_j + 1]
    probs_j = torch.softmax(logits_j, dim=-1)
    
    return (probs_i[aa_indices].cpu().numpy(),
            probs_j[aa_indices].cpu().numpy())


@torch.no_grad()
def compute_epistatic_coupling_map(seq, model, alphabet, batch_converter,
                                    device, n_sample_pairs, rng):
    L = len(seq)
    
    all_pairs = list(combinations(range(L), 2))
    n_pairs = min(n_sample_pairs, len(all_pairs))
    sampled_pairs = rng.sample(all_pairs, n_pairs)
    
    unique_positions = set()
    for i, j in sampled_pairs:
        unique_positions.add(i)
        unique_positions.add(j)
    
    single_marginals = compute_single_masked_marginals(
        seq, model, alphabet, batch_converter, device,
        sorted(unique_positions))
    
    coupling_scores = []
    for i, j in sampled_pairs:
        double_i, double_j = compute_double_masked_marginals(
            seq, model, alphabet, batch_converter, device, i, j)
        
        jsd_i = jensenshannon(single_marginals[i], double_i)
        jsd_j = jensenshannon(single_marginals[j], double_j)
        
        coupling = (jsd_i + jsd_j) / 2
        
        if not np.isnan(coupling):
            coupling_scores.append((i, j, coupling))

    coupling_scores.sort(key=lambda x: x[2], reverse=True)
    
    return coupling_scores


def build_coupling_groups(coupling_scores, L, n_groups, group_size_min,
                          group_size_max, threshold, rng):
    effective_threshold = threshold
    if coupling_scores:
        all_strengths = [s for _, _, s in coupling_scores]
        n_above = sum(1 for s in all_strengths if s >= threshold)
        frac_above = n_above / len(all_strengths)
        
        if frac_above < COUPLING_MIN_FRAC:
            # Lower threshold to capture top COUPLING_MIN_FRAC of pairs
            percentile_idx = int(len(all_strengths) * (1 - COUPLING_MIN_FRAC))
            sorted_strengths = sorted(all_strengths, reverse=True)
            effective_threshold = sorted_strengths[min(percentile_idx,
                                                       len(sorted_strengths) - 1)]
            effective_threshold = max(effective_threshold, 0.01)
    
    adjacency = {i: [] for i in range(L)}
    for i, j, strength in coupling_scores:
        if strength >= effective_threshold:
            adjacency[i].append((j, strength))
            adjacency[j].append((i, strength))
    
    coupling_count = {i: len(neighbors) for i, neighbors in adjacency.items()}
    
    used = set()
    groups = []
    sorted_positions = sorted(range(L), key=lambda x: coupling_count[x],
                               reverse=True)
    
    for seed_pos in sorted_positions:
        if seed_pos in used:
            continue
        if len(groups) >= n_groups:
            break
        if coupling_count[seed_pos] == 0:
            break 
        
        group = [seed_pos]
        used.add(seed_pos)
        
        neighbors = sorted(adjacency[seed_pos], key=lambda x: x[1],
                          reverse=True)
        for nb, _ in neighbors:
            if nb not in used and len(group) < group_size_max:
                group.append(nb)
                used.add(nb)
        
        # Pad with random unused positions if too small
        while len(group) < group_size_min:
            remaining = [p for p in range(L) if p not in used]
            if not remaining:
                break
            p = rng.choice(remaining)
            group.append(p)
            used.add(p)
        
        if len(group) >= group_size_min:
            groups.append(group)
    
    while len(groups) < n_groups:
        remaining = [p for p in range(L) if p not in used]
        if len(remaining) < group_size_min:
            break
        group = remaining[:group_size_min]
        for p in group:
            used.add(p)
        groups.append(group)
    
    return groups, effective_threshold


def propose_single(seq, k_mutations, rng):
    L = len(seq)
    positions = rng.sample(range(L), k_mutations)
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def propose_coupled(seq, groups, group_size_min, rng):
    if not groups:
        return propose_single(seq, 1, rng)
    
    group = rng.choice(groups)
    positions = list(group)
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def compute_adaptive_lambda_max(coupling_scores, threshold):
    above = [s for _, _, s in coupling_scores if s >= threshold]
    if not above:
        return LAMBDA_MIN, 0.0
    
    mean_coupling = np.mean(above)
    lambda_max = np.clip(mean_coupling * LAMBDA_COUPLING_SCALE,
                         LAMBDA_MIN, LAMBDA_CAP)
    return lambda_max, mean_coupling


def compute_lambda(current_prcs, start_prcs, lambda_max,
                   ramp_frac=LAMBDA_RAMP_FRAC):
    gap = 1.0 - start_prcs
    if gap <= 0:
        return lambda_max
    
    progress = (current_prcs - start_prcs) / gap  # 0 to 1
    ramp = min(1.0, max(0.0, progress / ramp_frac))
    return lambda_max * ramp



def run_walk(wt: str, protein_name: str, run_id: int, cfg: dict,
             emb_wt, model, alphabet, batch_converter, device,
             rng: random.Random) -> dict:
    L = len(wt)
    max_steps = cfg["max_steps"]
    k_single = cfg["k_single"]
    p_coupled = cfg["p_coupled"]
    n_groups = cfg["n_groups"]
    gsmin = cfg["group_size_min"]
    gsmax = cfg["group_size_max"]
    cooling = cfg["cooling_rate"]
    
    print(f"  [Run {run_id}] L={L}  k_single={k_single}  "
          f"group_size={gsmin}-{gsmax}  p_coupled={p_coupled:.2f}  "
          f"steps={max_steps}")
    
    current_seq = random_sequence(L, rng)
    current_emb = get_per_residue_embedding(current_seq, model, alphabet,
                                             batch_converter, device)
    
    start_prcs, start_perpos = score_prcs(current_emb, emb_wt)
    current_prcs = start_prcs
    current_combined = start_prcs  # λ=0 at start
    current_hamming = hamming(current_seq, wt)
    
    print(f"  [Run {run_id}] Start PRCS={start_prcs:.4f}  "
          f"Hamming={current_hamming}")
    
    n_sample_pairs = get_coupling_sample_pairs(L)
    print(f"  [Run {run_id}] Computing initial coupling map "
          f"(double-masking, {n_sample_pairs} pairs)...")
    coupling_scores = compute_epistatic_coupling_map(
        current_seq, model, alphabet, batch_converter, device,
        n_sample_pairs, rng)
    n_above_thresh = sum(1 for _, _, s in coupling_scores
                         if s >= COUPLING_THRESHOLD)
    print(f"  [Run {run_id}] {n_above_thresh}/{len(coupling_scores)} pairs "
          f"above coupling threshold {COUPLING_THRESHOLD}")
    groups, effective_threshold = build_coupling_groups(
        coupling_scores, L, n_groups, gsmin, gsmax,
        COUPLING_THRESHOLD, rng)
    group_sizes = [len(g) for g in groups]
    print(f"  [Run {run_id}] {len(groups)} coupled groups, "
          f"sizes: {group_sizes[:10]}{'...' if len(group_sizes) > 10 else ''}")
    if effective_threshold != COUPLING_THRESHOLD:
        print(f"  [Run {run_id}] Threshold adapted: {COUPLING_THRESHOLD} → "
              f"{effective_threshold:.4f}")

    lambda_max, mean_coupling = compute_adaptive_lambda_max(
        coupling_scores, effective_threshold)
    print(f"  [Run {run_id}] Adaptive λ_max={lambda_max:.3f}  "
          f"(mean coupling={mean_coupling:.4f})")
    
    temperature = INITIAL_TEMP
    best_seq = current_seq
    best_prcs = current_prcs
    best_hamming = current_hamming
    best_combined = current_combined
    
    trajectory = []
    n_accepted_single = 0
    n_proposed_single = 0
    n_accepted_coupled = 0
    n_proposed_coupled = 0
    
    for step in range(max_steps):
        lam = compute_lambda(current_prcs, start_prcs, lambda_max)
        
        if step > 0 and step % COUPLING_RECOMPUTE_EVERY == 0:
            print(f"  [Run {run_id}] Step {step}: recomputing coupling map "
                  f"(double-masking)...")
            coupling_scores = compute_epistatic_coupling_map(
                current_seq, model, alphabet, batch_converter, device,
                n_sample_pairs, rng)

            groups, effective_threshold = build_coupling_groups(
                coupling_scores, L, n_groups, gsmin, gsmax,
                COUPLING_THRESHOLD, rng)

            lambda_max, mean_coupling = compute_adaptive_lambda_max(
                coupling_scores, effective_threshold)
        
        candidates = []
        for _ in range(PROPOSALS_PER_STEP):
            if rng.random() < p_coupled and groups:
                cand_seq, cand_pos = propose_coupled(
                    current_seq, groups, gsmin, rng)
                is_coupled = True
            else:
                cand_seq, cand_pos = propose_single(
                    current_seq, k_single, rng)
                is_coupled = False
            candidates.append((cand_seq, cand_pos, is_coupled))
        
        best_cand = None
        best_cand_score = -999
        best_cand_prcs = -999
        best_cand_epi = 0
        best_cand_is_coupled = False
        
        for cand_seq, cand_pos, is_coupled in candidates:
            cand_emb = get_per_residue_embedding(
                cand_seq, model, alphabet, batch_converter, device)
            comb, prcs, epi, _ = compute_combined_score(
                cand_emb, emb_wt, groups, lam)
            
            if comb > best_cand_score:
                best_cand = (cand_seq, cand_emb)
                best_cand_score = comb
                best_cand_prcs = prcs
                best_cand_epi = epi
                best_cand_is_coupled = is_coupled
            
            if is_coupled:
                n_proposed_coupled += 1
            else:
                n_proposed_single += 1
        
        delta = best_cand_score - current_combined
        if delta > 0:
            accept = True
        elif temperature > 1e-10:
            accept = rng.random() < math.exp(delta / temperature)
        else:
            accept = False
        
        if accept:
            current_seq, current_emb = best_cand
            current_prcs = best_cand_prcs
            current_combined = best_cand_score
            current_hamming = hamming(current_seq, wt)
            
            if best_cand_is_coupled:
                n_accepted_coupled += 1
            else:
                n_accepted_single += 1
            
            if current_prcs > best_prcs:
                best_prcs = current_prcs
                best_seq = current_seq
                best_hamming = current_hamming
                best_combined = current_combined
        
        temperature *= cooling
        
        row = {
            "step": step,
            "prcs": current_prcs,
            "epi": best_cand_epi,
            "combined": current_combined,
            "hamming": current_hamming,
            "best_hamming": best_hamming,
            "best_prcs": best_prcs,
            "lambda": lam,
            "lambda_max": lambda_max,
            "temperature": temperature,
            "accepted": accept,
            "coupled": best_cand_is_coupled,
            "delta": delta,
        }
        trajectory.append(row)

        if step > 0 and step % LOG_EVERY == 0:
            msg = (f"  [Run {run_id}] Step {step}  "
                   f"PRCS={current_prcs:.4f}  Epi={best_cand_epi:.4f}  "
                   f"Combined={current_combined:.4f}  "
                   f"Hamming={current_hamming}  "
                   f"λ={lam:.3f}(max={lambda_max:.3f})  T={temperature:.4f}")
            print(msg)
        
        if step > 0 and step % PLL_DIAGNOSTIC_EVERY == 0:
            pll = score_true_pll(current_seq, model, alphabet,
                                  batch_converter, device)
            print(f"  [Run {run_id}] Step {step} PLL diagnostic: {pll:.4f}")
            trajectory[-1]["pll"] = pll
    
    acc_single = (n_accepted_single / n_proposed_single * 100
                  if n_proposed_single > 0 else 0)
    acc_coupled = (n_accepted_coupled / n_proposed_coupled * 100
                   if n_proposed_coupled > 0 else 0)
    
    final_hamming = hamming(current_seq, wt)
    
    print(f"  [Run {run_id}] Finished after {max_steps} steps")
    print(f"    Final PRCS={current_prcs:.4f}  Best PRCS={best_prcs:.4f}")
    print(f"    Final Hamming={final_hamming}  Best Hamming={best_hamming}")
    print(f"    Accept rate: single={acc_single:.1f}%  "
          f"coupled={acc_coupled:.1f}%")
    print(f"    Final λ={lam:.3f}  λ_max={lambda_max:.3f}")
    
    summary = {
        "protein": protein_name,
        "length": L,
        "run": run_id,
        "start_prcs": start_prcs,
        "start_hamming": hamming(random_sequence(L, rng), wt),  # approx
        "best_prcs": best_prcs,
        "best_hamming": best_hamming,
        "best_combined": best_combined,
        "final_prcs": current_prcs,
        "final_hamming": final_hamming,
        "final_combined": current_combined,
        "final_lambda": lam,
        "final_lambda_max": lambda_max,
        "accept_rate_single": acc_single,
        "accept_rate_coupled": acc_coupled,
        "steps": max_steps,
        "k_single": k_single,
        "p_coupled": p_coupled,
        "n_groups": len(groups),
    }
    
    return {
        "trajectory": trajectory,
        "summary": summary,
        "best_seq": best_seq,
        "final_seq": current_seq,
    }



def plot_walk(trajectories, protein_name, L, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{protein_name} (L={L}) — Evolution Walk v4",
                 fontsize=14, fontweight="bold")
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
    
    for idx, (traj, color) in enumerate(zip(trajectories, colors)):
        df = pd.DataFrame(traj)
        steps = df["step"]
        
        # PRCS trajectory
        axes[0, 0].plot(steps, df["prcs"], color=color, alpha=0.8,
                        label=f"Run {idx}")
        axes[0, 0].set_ylabel("PRCS")
        axes[0, 0].set_title("PRCS Trajectory")
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(steps, df["hamming"], color=color, alpha=0.8)
        axes[0, 1].plot(steps, df["best_hamming"], color=color, alpha=0.4,
                        linestyle="--")
        axes[0, 1].set_ylabel("Hamming to WT")
        axes[0, 1].set_title("Hamming Distance (solid=current, dash=best)")

        axes[0, 2].plot(steps, df["combined"], color=color, alpha=0.8)
        axes[0, 2].set_ylabel("Combined Score")
        axes[0, 2].set_title("Combined (PRCS + λ·Epi)")

        axes[1, 0].plot(steps, df["lambda"], color=color, alpha=0.8)
        if "lambda_max" in df.columns:
            axes[1, 0].plot(steps, df["lambda_max"], color=color, alpha=0.3,
                           linestyle="--")
        axes[1, 0].set_ylabel("λ")
        axes[1, 0].set_title("Lambda (solid=current, dash=λ_max)")
        axes[1, 0].set_xlabel("Step")

        axes[1, 1].plot(steps, df["epi"], color=color, alpha=0.8)
        axes[1, 1].set_ylabel("Epistatic Score")
        axes[1, 1].set_title("Epistatic Component")
        axes[1, 1].set_xlabel("Step")

        axes[1, 2].plot(steps, df["temperature"], color=color, alpha=0.8)
        axes[1, 2].set_ylabel("Temperature")
        axes[1, 2].set_title("Annealing Schedule")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_yscale("log")
    
    plt.tight_layout()
    path = outdir / "plots_walk.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_proposals(trajectories, protein_name, L, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{protein_name} — Proposal Analysis", fontsize=13)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
    
    for idx, (traj, color) in enumerate(zip(trajectories, colors)):
        df = pd.DataFrame(traj)
        
        window = 50
        if len(df) > window:
            rolling_acc = df["accepted"].rolling(window).mean() * 100
            axes[0].plot(df["step"], rolling_acc, color=color, alpha=0.8,
                        label=f"Run {idx}")
        axes[0].set_ylabel("Acceptance Rate (%)")
        axes[0].set_title(f"Rolling Acceptance (window={window})")
        axes[0].legend(fontsize=8)
        
        deltas = df["delta"].dropna()
        axes[1].hist(deltas[deltas > 0], bins=50, alpha=0.4, color="green",
                     label="Uphill" if idx == 0 else None)
        axes[1].hist(deltas[deltas <= 0], bins=50, alpha=0.4, color="red",
                     label="Downhill" if idx == 0 else None)
        axes[1].set_title("Score Delta Distribution")
        axes[1].legend(fontsize=8)

    all_coupled_acc = []
    all_single_acc = []
    for traj in trajectories:
        df = pd.DataFrame(traj)
        coupled = df[df["coupled"] == True]
        single = df[df["coupled"] == False]
        if len(coupled) > 0:
            all_coupled_acc.append(coupled["accepted"].mean() * 100)
        if len(single) > 0:
            all_single_acc.append(single["accepted"].mean() * 100)
    
    labels = ["Single", "Coupled"]
    means = [np.mean(all_single_acc) if all_single_acc else 0,
             np.mean(all_coupled_acc) if all_coupled_acc else 0]
    bars = axes[2].bar(labels, means, color=["steelblue", "coral"])
    axes[2].set_ylabel("Acceptance Rate (%)")
    axes[2].set_title("Single vs Coupled Proposals")
    for bar, val in zip(bars, means):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=10)
    
    plt.tight_layout()
    path = outdir / "plots_proposals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_cross_protein(all_results, outdir):
    if not all_results:
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Cross-Protein Comparison — v10b-v3", fontsize=14,
                 fontweight="bold")
    
    proteins = []
    lengths = []
    best_hammings = []
    pct_recovered = []
    best_prcs_vals = []
    coupled_advantages = []
    
    for res in all_results:
        proteins.append(res["protein"])
        lengths.append(res["length"])
        best_hammings.append(res["best_hamming"])
        pct_recovered.append(
            100 * (1 - res["best_hamming"] / res["length"]))
        best_prcs_vals.append(res["best_prcs"])
        sa = res.get("accept_rate_single", 0)
        ca = res.get("accept_rate_coupled", 0)
        coupled_advantages.append(ca / sa if sa > 0 else 0)
    
    x = range(len(proteins))
    
    axes[0].bar(x, best_hammings, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(proteins, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Best Hamming")
    axes[0].set_title("Best Hamming Distance")
    
    axes[1].bar(x, pct_recovered, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(proteins, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("% Residues Recovered")
    axes[1].set_title("Sequence Recovery")
    axes[1].axhline(y=82, color="green", linestyle="--", alpha=0.5,
                     label="Insulin v10b (82%)")
    axes[1].legend(fontsize=8)
    
    axes[2].bar(x, best_prcs_vals, color="mediumpurple")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(proteins, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Best PRCS")
    axes[2].set_title("Best PRCS Achieved")
    
    axes[3].scatter(lengths, pct_recovered, s=100, c="steelblue",
                     edgecolors="black")
    for i, name in enumerate(proteins):
        axes[3].annotate(name, (lengths[i], pct_recovered[i]),
                         fontsize=7, ha="center", va="bottom")
    axes[3].set_xlabel("Protein Length")
    axes[3].set_ylabel("% Residues Recovered")
    axes[3].set_title("Length vs Recovery")
    
    plt.tight_layout()
    path = Path(outdir) / "plots_cross_protein.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    top_outdir = Path(OUTDIR)
    top_outdir.mkdir(exist_ok=True, parents=True)
    
    # Discover FASTA files
    fasta_dir = Path(FASTA_DIR)
    if not fasta_dir.exists():
        print(f"ERROR: FASTA directory '{FASTA_DIR}' not found.")
        sys.exit(1)
    
    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        print(f"ERROR: No .fasta or .fa files found in '{FASTA_DIR}'")
        sys.exit(1)
    
    proteins = []
    for fp in fasta_files:
        name, seq = read_fasta_one(str(fp))
        # Use filename stem as protein name for consistency
        name = fp.stem
        proteins.append({"name": name, "seq": seq, "length": len(seq),
                         "path": str(fp)})

    print("=" * 70)
    print("  EVOLUTION WALK V4 — MULTI-PROTEIN")
    print(f"  FASTA directory: {FASTA_DIR}")
    print(f"  Proteins: {len(proteins)}")
    print(f"  Runs per protein: {RUNS_PER_PROTEIN}")
    print(f"  Steps: length-adaptive (max(2000, 10*L)), NO early stopping")
    print(f"  Coupling: double-masking (length-adaptive pairs, "
          f"threshold={COUPLING_THRESHOLD}, adaptive λ_max)")
    print("=" * 70)
    for p in proteins:
        cfg = get_protein_config(p["length"])
        n_pairs = get_coupling_sample_pairs(p["length"])
        print(f"  {p['name']:40s} {p['length']:4d} residues  "
              f"k={cfg['k_single']}  steps={cfg['max_steps']}  "
              f"p_coupled={cfg['p_coupled']:.2f}  pairs={n_pairs}")
    print()
    
    model, alphabet, batch_converter = load_esm2(DEVICE)
    
    all_cross_results = []
    
    for pidx, protein in enumerate(proteins):
        pname = protein["name"]
        wt = protein["seq"]
        L = protein["length"]
        cfg = get_protein_config(L)
        
        prot_outdir = top_outdir / pname
        prot_outdir.mkdir(exist_ok=True, parents=True)
        
        print()
        print("#" * 70)
        print(f"  PROTEIN {pidx + 1}/{len(proteins)}: {pname}")
        print(f"  Length: {L}  Steps: {cfg['max_steps']}  "
              f"k_single: {cfg['k_single']}  p_coupled: {cfg['p_coupled']:.2f}")
        print(f"  Groups: {cfg['n_groups']} of size "
              f"{cfg['group_size_min']}-{cfg['group_size_max']}")
        print("#" * 70)
        
        print("  Computing WT reference embedding...")
        emb_wt = get_per_residue_embedding(wt, model, alphabet,
                                            batch_converter, DEVICE)
        
        test_rng = random.Random(0)
        test_seq = random_sequence(L, test_rng)
        test_emb = get_per_residue_embedding(test_seq, model, alphabet,
                                              batch_converter, DEVICE)
        rand_prcs, _ = score_prcs(test_emb, emb_wt)
        rand_hamming = hamming(test_seq, wt)
        print(f"  WT PRCS: 1.000000")
        print(f"  Random PRCS: {rand_prcs:.4f}  Random Hamming: {rand_hamming}")
        
        all_trajectories = []
        all_summaries = []
        best_overall_hamming = L + 1
        best_overall_seq = None
        best_overall_prcs = -999
        
        for run_id in range(RUNS_PER_PROTEIN):
            print()
            print(f"  {'=' * 55}")
            print(f"    RUN {run_id + 1}/{RUNS_PER_PROTEIN}  [{pname}]")
            print(f"  {'=' * 55}")
            
            run_rng = random.Random(SEED + pidx * 100 + run_id)
            t0 = time.time()
            
            result = run_walk(wt, pname, run_id, cfg, emb_wt,
                              model, alphabet, batch_converter, DEVICE,
                              run_rng)
            
            dt = time.time() - t0
            s = result["summary"]
            print(f"    Run {run_id} done in {dt / 60:.1f} min  "
                  f"PRCS={s['best_prcs']:.4f}  "
                  f"Best Hamming={s['best_hamming']}  "
                  f"Final Hamming={s['final_hamming']}")
            
            all_trajectories.append(result["trajectory"])
            all_summaries.append(s)
            
            if s["best_hamming"] < best_overall_hamming:
                best_overall_hamming = s["best_hamming"]
                best_overall_seq = result["best_seq"]
                best_overall_prcs = s["best_prcs"]
        
        print(f"\n  [{pname}] SUMMARY:")
        for s in all_summaries:
            print(f"    Run {s['run']}: PRCS={s['best_prcs']:.4f}  "
                  f"Hamming={s['final_hamming']} (best={s['best_hamming']})")
        print(f"    Mean best Hamming: "
              f"{np.mean([s['best_hamming'] for s in all_summaries]):.1f}")
        print(f"    Best overall Hamming: {best_overall_hamming}")
        
        for run_id, traj in enumerate(all_trajectories):
            df = pd.DataFrame(traj)
            df.to_csv(prot_outdir / f"trajectory_run{run_id}.csv",
                      index=False)
        
        df_summ = pd.DataFrame(all_summaries)
        df_summ.to_csv(prot_outdir / "summary.csv", index=False)
        
        if best_overall_seq:
            with open(prot_outdir / "best_sequence.fasta", "w") as f:
                f.write(f">{pname}_v10b_v3_best_H{best_overall_hamming}\n")
                for i in range(0, len(best_overall_seq), 60):
                    f.write(best_overall_seq[i:i + 60] + "\n")
        
        plot_walk(all_trajectories, pname, L, prot_outdir)
        plot_proposals(all_trajectories, pname, L, prot_outdir)
        
        best_summary = min(all_summaries, key=lambda s: s["best_hamming"])
        all_cross_results.append({
            "protein": pname,
            "length": L,
            "best_hamming": best_overall_hamming,
            "best_prcs": best_overall_prcs,
            "mean_best_hamming": np.mean(
                [s["best_hamming"] for s in all_summaries]),
            "accept_rate_single": np.mean(
                [s["accept_rate_single"] for s in all_summaries]),
            "accept_rate_coupled": np.mean(
                [s["accept_rate_coupled"] for s in all_summaries]),
            "k_single": cfg["k_single"],
            "p_coupled": cfg["p_coupled"],
            "steps": cfg["max_steps"],
        })

    print("\n" + "=" * 70)
    print("  CROSS-PROTEIN SUMMARY")
    print("=" * 70)
    print(f"  {'Protein':<25s} {'L':>4s} {'Best H':>7s} {'%Rec':>6s} "
          f"{'PRCS':>6s} {'k':>3s} {'Steps':>6s}")
    print(f"  {'-' * 25} {'-' * 4} {'-' * 7} {'-' * 6} {'-' * 6} "
          f"{'-' * 3} {'-' * 6}")
    for r in all_cross_results:
        pct = 100 * (1 - r["best_hamming"] / r["length"])
        print(f"  {r['protein']:<25s} {r['length']:4d} "
              f"{r['best_hamming']:7d} {pct:5.1f}% "
              f"{r['best_prcs']:6.3f} {r['k_single']:3d} "
              f"{r['steps']:6d}")
    
    # Save cross-protein CSV
    df_cross = pd.DataFrame(all_cross_results)
    df_cross.to_csv(top_outdir / "cross_protein_summary.csv", index=False)
    
    # Cross-protein plot
    plot_cross_protein(all_cross_results, top_outdir)
    
    print(f"\nAll results saved to {top_outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
