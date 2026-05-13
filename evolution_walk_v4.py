"""
Evolution walk v4: Per-residue cosine similarity scoring.


Replace PLL with per-residue cosine similarity (PRCS):

    score(seq) = (1/L) * sum_i cos(h_i^seq, h_i^WT)

where h_i^seq is ESM2's per-residue embedding at position i for the
candidate sequence, and h_i^WT is the same for wild-type insulin.

ADDITIONAL METRICS TRACKED

- PLL: still computed every N steps as a diagnostic (not for decisions)
- EDS: L2 embedding distance (our old metric, for comparison)
- Hamming distance to WT
- Per-position cosine similarity breakdown

WALK DESIGN

- MCMC with simulated annealing
- Mixed proposals: 40% coupled (from coupling map), 60% single-point
- Coupling map from JSD of conditional distributions 
- Multiple runs from random starts
"""

import os
import warnings
from pathlib import Path
import random
import time
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*libomp.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import esm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


WT_FASTA            = "insulin_fastas/insulin_human_wt.fasta"
OUTDIR              = "evolution_walk_v10b"

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
SEED                = 42
REPR_LAYER          = 33

N_RUNS              = 20
MAX_STEPS           = 1000
START_FROM_RANDOM   = True

T_INIT              = 0.3
T_MIN               = 0.003     
COOLING             = 0.997     

COUPLING_TOP_K      = 5
GROUP_SIZE_MIN      = 3
GROUP_SIZE_MAX      = 5
N_GROUPS            = 15
RECOMPUTE_INTERVAL  = 75      

P_COUPLED           = 0.4
N_CANDIDATES_SINGLE = 30     
N_CANDIDATES_COUPLED = 10

PLL_INTERVAL        = 200       
SNAPSHOT_INTERVAL   = 25       
PLL_POS_BATCH       = 16

LAMBDA_EPISTASIS    = 0.3  
EPISTASIS_MAX_K     = 5       

AMINO_ACIDS         = list("ACDEFGHIKLMNPQRSTVWY")


def read_fasta_one(path):
    txt = Path(path).read_text().strip().splitlines()
    return "".join(ln.strip() for ln in txt if not ln.startswith(">")).replace(" ", "").upper()


def mutate(seq, changes):
    s = list(seq)
    for pos, aa in changes.items():
        s[pos] = aa
    return "".join(s)


def random_sequence(length):
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def load_esm2(device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(torch.device(device))
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def write_fasta(path, name, seq):
    with open(path, "w") as f:
        f.write(f">{name}\n{seq}\n")


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
        return 0.0, {}

    total = 0.0
    n_pairs = 0
    group_scores = {}

    for g_idx, group in enumerate(groups):
        k = min(len(group), EPISTASIS_MAX_K)
        positions = group[:k]
        group_score = 0.0
        group_pairs = 0

        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                i, j = positions[a], positions[b]
                # Difference vectors
                diff_cand = emb_candidate[i] - emb_candidate[j]  # (d_model,)
                diff_wt   = emb_wt[i] - emb_wt[j]                # (d_model,)
                # Cosine similarity of the difference vectors
                cos_val = F.cosine_similarity(
                    diff_cand.unsqueeze(0), diff_wt.unsqueeze(0)
                ).item()
                group_score += cos_val
                group_pairs += 1

        if group_pairs > 0:
            group_score /= group_pairs
            group_scores[g_idx] = group_score
            total += group_score
            n_pairs += 1

    mean_score = total / max(1, n_pairs)
    return mean_score, group_scores


def score_combined(emb_candidate, emb_wt, groups):

    prcs_score, per_pos = score_prcs(emb_candidate, emb_wt)
    epi_score, group_scores = score_epistatic(emb_candidate, emb_wt, groups)

    combined = prcs_score + LAMBDA_EPISTASIS * epi_score

    return combined, prcs_score, epi_score, per_pos, group_scores


def score_eds(emb_candidate, emb_wt):
    return -torch.norm(emb_candidate - emb_wt).item()


@torch.no_grad()
def score_true_pll(seq, model, alphabet, batch_converter, device):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    mask_idx = alphabet.mask_idx
    L = len(seq)
    pll = 0.0
    for start in range(0, L, PLL_POS_BATCH):
        end = min(L, start + PLL_POS_BATCH)
        positions = list(range(start, end))
        bsz = len(positions)
        batch_tokens = tokens.repeat(bsz, 1)
        for r, pos in enumerate(positions):
            batch_tokens[r, pos + 1] = mask_idx
        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        log_probs = torch.log_softmax(out["logits"], dim=-1)
        for r, pos in enumerate(positions):
            true_idx = tokens[0, pos + 1]
            pll += log_probs[r, pos + 1, true_idx].item()
    return float(pll / L)


@torch.no_grad()
def compute_coupling_map(seq, model, alphabet, batch_converter, device):
    L = len(seq)
    data = [("seq", seq)]
    _, _, tokens_full = batch_converter(data)
    tokens_full = tokens_full.to(device)
    mask_idx = alphabet.mask_idx
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]

    out_full = model(tokens_full, repr_layers=[], return_contacts=False)
    logits_full = out_full["logits"][0]
    probs_full = torch.softmax(logits_full, dim=-1)
    baseline = probs_full[1:L+1][:, aa_indices].cpu().numpy()

    coupling = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        tokens_masked = tokens_full.clone()
        tokens_masked[0, i + 1] = mask_idx
        out_masked = model(tokens_masked, repr_layers=[], return_contacts=False)
        logits_masked = out_masked["logits"][0]
        probs_masked = torch.softmax(logits_masked, dim=-1)
        shifted = probs_masked[1:L+1][:, aa_indices].cpu().numpy()

        for j in range(L):
            if j == i:
                continue
            jsd = jensenshannon(baseline[j], shifted[j])
            if np.isfinite(jsd):
                coupling[i, j] = jsd

    return coupling


def extract_coupled_groups(coupling, n_groups, group_size_min, group_size_max):
    L = coupling.shape[0]
    groups = []
    total_coupling = coupling.sum(axis=1) + coupling.sum(axis=0)
    ranked = np.argsort(-total_coupling)
    used = set()

    for center in ranked:
        if center in used:
            continue
        mutual = coupling[center, :] + coupling[:, center]
        mutual[center] = -1
        partners = np.argsort(-mutual)
        group = [center]
        for p in partners:
            if len(group) >= group_size_max:
                break
            if p not in used:
                group.append(int(p))
        if len(group) >= group_size_min:
            groups.append(sorted(group))
            used.add(center)
        if len(groups) >= n_groups:
            break

    return groups[:n_groups]


def propose_coupled(seq, group, rng):
    changes = {}
    for pos in group:
        aa = rng.choice(AMINO_ACIDS)
        while aa == seq[pos]:
            aa = rng.choice(AMINO_ACIDS)
        changes[pos] = aa
    return changes


def propose_single(seq, L, rng):
    pos = rng.randint(0, L - 1)
    aa = rng.choice(AMINO_ACIDS)
    while aa == seq[pos]:
        aa = rng.choice(AMINO_ACIDS)
    return {pos: aa}


def run_walk(run_idx, wt, emb_wt, model, alphabet, batch_converter, device, rng,
             outdir):
    L = len(wt)
    seq = random_sequence(L) if START_FROM_RANDOM else wt

    emb_current = get_per_residue_embedding(seq, model, alphabet, batch_converter, device)
    current_prcs, current_perpos = score_prcs(emb_current, emb_wt)
    T = T_INIT

    print(f"  [Run {run_idx}] Computing initial coupling map...")
    coupling = compute_coupling_map(seq, model, alphabet, batch_converter, device)
    groups = extract_coupled_groups(coupling, N_GROUPS, GROUP_SIZE_MIN, GROUP_SIZE_MAX)
    print(f"  [Run {run_idx}] {len(groups)} coupled groups, sizes: {[len(g) for g in groups]}")

    current_combined, current_prcs, current_epi, current_perpos, _ = \
        score_combined(emb_current, emb_wt, groups)

    trajectory = []
    snapshots_perpos = [] 
    best_seq = seq
    best_prcs = current_prcs
    best_hamming = hamming(seq, wt)

    n_coupled_proposed = 0
    n_coupled_accepted = 0
    n_single_proposed = 0
    n_single_accepted = 0
    coupled_deltas = []
    single_deltas = []

    for step in range(MAX_STEPS):
        use_coupled = rng.random() < P_COUPLED and len(groups) > 0

        best_candidate_seq = None
        best_candidate_emb = None
        best_candidate_score = -float("inf")
        best_candidate_prcs = None
        best_candidate_epi = None
        proposal_type = None

        if use_coupled:
            for _ in range(N_CANDIDATES_COUPLED):
                group = rng.choice(groups)
                changes = propose_coupled(seq, group, rng)
                cand = mutate(seq, changes)
                emb_cand = get_per_residue_embedding(cand, model, alphabet, batch_converter, device)
                cand_combined, cand_prcs, cand_epi, _, _ = \
                    score_combined(emb_cand, emb_wt, groups)
                n_coupled_proposed += 1
                if cand_combined > best_candidate_score:
                    best_candidate_score = cand_combined
                    best_candidate_prcs = cand_prcs
                    best_candidate_epi = cand_epi
                    best_candidate_seq = cand
                    best_candidate_emb = emb_cand
                    proposal_type = "coupled"
        else:
            for _ in range(N_CANDIDATES_SINGLE):
                changes = propose_single(seq, L, rng)
                cand = mutate(seq, changes)
                emb_cand = get_per_residue_embedding(cand, model, alphabet, batch_converter, device)
                cand_combined, cand_prcs, cand_epi, _, _ = \
                    score_combined(emb_cand, emb_wt, groups)
                n_single_proposed += 1
                if cand_combined > best_candidate_score:
                    best_candidate_score = cand_combined
                    best_candidate_prcs = cand_prcs
                    best_candidate_epi = cand_epi
                    best_candidate_seq = cand
                    best_candidate_emb = emb_cand
                    proposal_type = "single"

        delta = best_candidate_score - current_combined
        if delta > 0:
            accept = True
        else:
            accept = rng.random() < math.exp(delta / T) if T > 1e-10 else False

        if accept:
            seq = best_candidate_seq
            emb_current = best_candidate_emb
            current_combined = best_candidate_score
            current_prcs = best_candidate_prcs
            current_epi = best_candidate_epi
            if proposal_type == "coupled":
                n_coupled_accepted += 1
                coupled_deltas.append(delta)
            else:
                n_single_accepted += 1
                single_deltas.append(delta)

        h = hamming(seq, wt)
        if h < best_hamming:
            best_hamming = h
            best_seq = seq
            best_prcs = current_prcs

        T = max(T * COOLING, T_MIN)

        if step % SNAPSHOT_INTERVAL == 0:
            _, perpos = score_prcs(emb_current, emb_wt)
            snapshots_perpos.append({"step": step, "perpos": perpos.copy()})

        row = {
            "run": run_idx, "step": step,
            "combined": current_combined,
            "prcs": current_prcs, "epi": current_epi,
            "hamming": h,
            "accepted": accept, "proposal_type": proposal_type,
            "delta": delta, "temperature": T,
        }

        row["eds"] = score_eds(emb_current, emb_wt)

        if step % PLL_INTERVAL == 0:
            pll = score_true_pll(seq, model, alphabet, batch_converter, device)
            row["pll"] = pll
        else:
            row["pll"] = None

        trajectory.append(row)

        if (step + 1) % RECOMPUTE_INTERVAL == 0 and step < MAX_STEPS - 1:
            print(f"  [Run {run_idx}] Step {step+1}: recomputing coupling map...")
            coupling = compute_coupling_map(seq, model, alphabet, batch_converter, device)
            groups = extract_coupled_groups(coupling, N_GROUPS, GROUP_SIZE_MIN, GROUP_SIZE_MAX)
            current_combined, current_prcs, current_epi, _, _ = \
                score_combined(emb_current, emb_wt, groups)

        if (step + 1) % 50 == 0:
            print(f"  [Run {run_idx}] Step {step+1}/{MAX_STEPS}  "
                  f"PRCS={current_prcs:.4f}  Epi={current_epi:.4f}  "
                  f"Combined={current_combined:.4f}  Hamming={h}  T={T:.4f}")

    write_fasta(outdir / f"best_run{run_idx}.fasta", f"best_run{run_idx}_H{best_hamming}", best_seq)

    summary = {
        "run": run_idx,
        "final_prcs": current_prcs,
        "best_prcs": best_prcs,
        "final_hamming": h,
        "best_hamming": best_hamming,
        "start_hamming": trajectory[0]["hamming"],
        "n_coupled_proposed": n_coupled_proposed,
        "n_coupled_accepted": n_coupled_accepted,
        "n_single_proposed": n_single_proposed,
        "n_single_accepted": n_single_accepted,
        "coupled_accept_rate": n_coupled_accepted / max(1, n_coupled_proposed),
        "single_accept_rate": n_single_accepted / max(1, n_single_proposed),
        "mean_coupled_delta": np.mean(coupled_deltas) if coupled_deltas else 0,
        "mean_single_delta": np.mean(single_deltas) if single_deltas else 0,
    }

    return trajectory, summary, coupling, snapshots_perpos


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    wt = read_fasta_one(WT_FASTA)
    L = len(wt)
    print(f"WT length: {L}")

    print("Loading ESM2...")
    model, alphabet, batch_converter = load_esm2(DEVICE)

    print("Computing WT reference embedding...")
    emb_wt = get_per_residue_embedding(wt, model, alphabet, batch_converter, DEVICE)

    wt_prcs, wt_perpos = score_prcs(emb_wt, emb_wt)
    wt_pll = score_true_pll(wt, model, alphabet, batch_converter, DEVICE)
    print(f"WT PRCS: {wt_prcs:.6f} (should be 1.0)")
    print(f"WT PLL:  {wt_pll:.4f}")

    rand_seq = random_sequence(L)
    emb_rand = get_per_residue_embedding(rand_seq, model, alphabet, batch_converter, DEVICE)
    rand_prcs, _ = score_prcs(emb_rand, emb_wt)
    rand_eds = score_eds(emb_rand, emb_wt)
    print(f"Random PRCS: {rand_prcs:.4f}  (this is the floor)")
    print(f"Random EDS:  {rand_eds:.1f}")
    print(f"Random Hamming: {hamming(rand_seq, wt)}")

    all_trajectories = []
    all_summaries = []
    last_coupling = None
    all_snapshots = []

    for run_idx in range(N_RUNS):
        print(f"\n{'='*60}")
        print(f"  RUN {run_idx + 1}/{N_RUNS}")
        print(f"{'='*60}")
        t0 = time.time()

        traj, summary, coupling, snapshots = run_walk(
            run_idx, wt, emb_wt, model, alphabet, batch_converter,
            DEVICE, rng, outdir
        )
        all_trajectories.extend(traj)
        all_summaries.append(summary)
        last_coupling = coupling
        all_snapshots.append(snapshots)

        dt = time.time() - t0
        print(f"  Run {run_idx} done in {dt/60:.1f} min  "
              f"PRCS={summary['final_prcs']:.4f}  "
              f"Best Hamming={summary['best_hamming']}  "
              f"Final Hamming={summary['final_hamming']}")

    df_traj = pd.DataFrame(all_trajectories)
    df_traj.to_csv(outdir / "trajectory.csv", index=False)

    df_summ = pd.DataFrame(all_summaries)
    df_summ.to_csv(outdir / "summary.csv", index=False)

    if last_coupling is not None:
        np.save(outdir / "coupling_map.npy", last_coupling)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for _, row in df_summ.iterrows():
        print(f"  Run {int(row['run'])}: PRCS={row['final_prcs']:.4f}  "
              f"Hamming={int(row['final_hamming'])} (best={int(row['best_hamming'])})  "
              f"coupled_acc={row['coupled_accept_rate']:.3f}  "
              f"single_acc={row['single_accept_rate']:.3f}")
    print(f"\n  Mean best Hamming: {df_summ['best_hamming'].mean():.1f}")
    print(f"  Best overall Hamming: {df_summ['best_hamming'].min()}")

    make_plots(df_traj, df_summ, last_coupling, all_snapshots, wt, rand_prcs, outdir)


def make_plots(df_traj, df_summ, coupling, all_snapshots, wt, rand_prcs, outdir):
    L = len(wt)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    ax = axs[0, 0]
    for run in df_traj["run"].unique():
        sub = df_traj[df_traj["run"] == run]
        ax.plot(sub["step"], sub["prcs"], alpha=0.7, label=f"Run {run}")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.4, label="WT (1.0)")
    ax.axhline(rand_prcs, color="red", linestyle=":", alpha=0.4, label=f"Random ({rand_prcs:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("PRCS (per-residue cosine similarity)")
    ax.set_title("Primary score: PRCS trajectory")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axs[0, 1]
    for run in df_traj["run"].unique():
        sub = df_traj[df_traj["run"] == run]
        ax.plot(sub["step"], sub["hamming"], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Hamming distance to WT")
    ax.set_title("Hamming distance trajectory")
    ax.grid(alpha=0.3)

    ax = axs[1, 0]
    for run in df_traj["run"].unique():
        sub = df_traj[df_traj["run"] == run]
        ax.plot(sub["step"], sub["eds"], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("EDS (negative L2 distance)")
    ax.set_title("EDS trajectory (diagnostic)")
    ax.grid(alpha=0.3)

    ax = axs[1, 1]
    for run in df_traj["run"].unique():
        sub = df_traj[(df_traj["run"] == run) & df_traj["pll"].notna()]
        if len(sub) > 0:
            ax.plot(sub["step"], sub["pll"], "o-", alpha=0.7, markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("PLL (diagnostic)")
    ax.set_title("PLL trajectory (sampled every 50 steps)")
    ax.grid(alpha=0.3)

    fig.suptitle("V10: Per-residue cosine similarity walk", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "plots_walk.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'plots_walk.png'}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax = axs[0]
    x = np.arange(len(df_summ))
    w = 0.35
    ax.bar(x - w/2, df_summ["coupled_accept_rate"], w, label="Coupled", color="darkorange")
    ax.bar(x + w/2, df_summ["single_accept_rate"], w, label="Single", color="steelblue")
    ax.set_xlabel("Run")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Acceptance rates")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axs[1]
    ax.bar(x - w/2, df_summ["n_coupled_accepted"], w, label="Coupled", color="darkorange")
    ax.bar(x + w/2, df_summ["n_single_accepted"], w, label="Single", color="steelblue")
    ax.set_xlabel("Run")
    ax.set_ylabel("Accepted count")
    ax.set_title("Total accepted proposals")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axs[2]
    coupled_d = df_traj[df_traj["proposal_type"] == "coupled"]["delta"].dropna()
    single_d = df_traj[df_traj["proposal_type"] == "single"]["delta"].dropna()
    if len(coupled_d) > 0:
        ax.hist(coupled_d, bins=40, alpha=0.6, color="darkorange",
                label=f"Coupled (n={len(coupled_d)})", density=True)
    if len(single_d) > 0:
        ax.hist(single_d, bins=40, alpha=0.6, color="steelblue",
                label=f"Single (n={len(single_d)})", density=True)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Delta PRCS")
    ax.set_ylabel("Density")
    ax.set_title("PRCS change distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "plots_proposals.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'plots_proposals.png'}")

    if all_snapshots and len(all_snapshots[0]) > 0:
        snaps = all_snapshots[0]
        steps = [s["step"] for s in snaps]
        matrix = np.array([s["perpos"] for s in snaps]) 

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(matrix.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xlabel("Snapshot index (every 10 steps)")
        ax.set_ylabel("Position")
        ax.set_title("Per-position cosine similarity over time (Run 0)\n"
                     "Green = insulin-like, Red = diverged")
        plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
        fig.tight_layout()
        fig.savefig(outdir / "plots_perposition.png", dpi=150)
        plt.close(fig)
        print(f"Saved {outdir / 'plots_perposition.png'}")

    if coupling is not None:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        ax = axs[0]
        sym = (coupling + coupling.T) / 2
        im = ax.imshow(sym, cmap="hot", aspect="auto", origin="lower")
        ax.set_xlabel("Position j")
        ax.set_ylabel("Position i")
        ax.set_title("Coupling matrix (JSD)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax = axs[1]
        total = sym.sum(axis=1)
        ax.bar(range(L), total, color="steelblue", alpha=0.8)
        ax.set_xlabel("Position")
        ax.set_ylabel("Total coupling")
        ax.set_title("Per-position coupling strength")
        ax.grid(alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(outdir / "plots_coupling.png", dpi=150)
        plt.close(fig)
        print(f"Saved {outdir / 'plots_coupling.png'}")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    ax = axs[0]
    ax.bar(range(len(df_summ)), df_summ["best_prcs"], color="steelblue")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.3)
    ax.axhline(rand_prcs, color="red", linestyle=":", alpha=0.3)
    ax.set_xlabel("Run")
    ax.set_ylabel("Best PRCS")
    ax.set_title("Best PRCS achieved per run")
    ax.grid(alpha=0.3, axis="y")

    ax = axs[1]
    ax.bar(range(len(df_summ)), df_summ["best_hamming"], color="darkorange")
    ax.set_xlabel("Run")
    ax.set_ylabel("Best Hamming")
    ax.set_title("Best Hamming distance achieved")
    ax.grid(alpha=0.3, axis="y")

    ax = axs[2]
    reduction = df_summ["start_hamming"] - df_summ["best_hamming"]
    ax.bar(range(len(df_summ)), reduction, color="green")
    ax.set_xlabel("Run")
    ax.set_ylabel("Hamming reduction")
    ax.set_title("Hamming improvement\n(start − best)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("V10 Final results", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "plots_final.png", dpi=150)
    plt.close(fig)
    print(f"Saved {outdir / 'plots_final.png'}")


if __name__ == "__main__":
    main()
