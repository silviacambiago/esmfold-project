"""
Evolution walk v5: MCMC scoring strategy comparison.

All runs start from fully random sequences. Four scoring strategies compared:

  1. SOFT λ=0.01  — score = PLL + 0.01*EDS  (weak pull toward WT)
  2. SOFT λ=0.05  — score = PLL + 0.05*EDS  (moderate pull)
  3. SOFT λ=0.10  — score = PLL + 0.10*EDS  (strong pull)
  4. HARD CONSTRAINT — score = PLL, but reject any mutation that worsens
     EDS by more than EDS_HARD_THRESHOLD (structural floor)

Outputs:
- outdir/trajectory.csv          : per-step data
- outdir/summary.csv             : per-run summary
- outdir/best_*.fasta            : best sequence per strategy
- outdir/plots_comparison.png    : 2x2 head-to-head comparison
- outdir/plots_hamming.png       : 2x2 Hamming distance analysis
- outdir/plots_mcmc.png          : 3-panel MCMC diagnostics
- outdir/plots_eds.png           : 3-panel EDS analysis
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
OUTDIR = "evolution_walk_v5_scoring"

REPR_LAYER = 33
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Scoring strategies to compare
STRATEGIES = [
    {"name": "λ=0.01",  "mode": "combined", "lambda": 0.01, "color": "#3498db"},
    {"name": "λ=0.05",  "mode": "combined", "lambda": 0.05, "color": "#e67e22"},
    {"name": "λ=0.10",  "mode": "combined", "lambda": 0.10, "color": "#2ecc71"},
    {"name": "hard",     "mode": "hard",     "lambda": 0.0,  "color": "#9b59b6"},
]
RUNS_PER_STRATEGY = 5

# Hard constraint: reject mutations that worsen EDS by more than this
EDS_HARD_THRESHOLD = 0.5

MAX_STEPS = 500
PROPOSALS_PER_STEP = 30
TOP_K_REEVAL = 3

# MCMC
INITIAL_TEMP = 0.1
MIN_TEMP = 0.001
COOLING_RATE = 0.993

# Convergence
PATIENCE = 150
CONVERGENCE_WINDOW = 50
VAR_THRESHOLD = 1e-6

# EDS
EDS_TRACK_EVERY = 25

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


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


def propose_mutations(current, k, rng, entropies):
    L = len(current)
    w = np.clip(entropies, 1e-8, None); w /= w.sum()
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


def run_walk(start_seq, wt_seq, wt_emb, model, alphabet, bc, device, rng,
             scoring_mode="combined", eds_lambda=0.01, eds_hard_threshold=0.5):
    """
    scoring_mode:
      "combined" — score = PLL + lambda*EDS, MCMC on combined score
      "hard"     — score = PLL, but reject if EDS worsens beyond threshold
    """
    current = start_seq
    current_pll = score_true_pll(current, model, alphabet, bc, device)
    cur_emb = get_embeddings(current, model, bc, device)
    current_eds = compute_eds(cur_emb, wt_emb)

    def calc_score(pll, eds):
        if scoring_mode == "combined":
            return pll + eds_lambda * eds
        return pll  # hard mode uses pure PLL for scoring

    current_score = calc_score(current_pll, current_eds)
    best_seq, best_pll, best_eds = current, current_pll, current_eds
    best_score = current_score
    best_score_history = [best_score]
    steps_since_imp = 0
    accepted_up = 0; accepted_down = 0
    temp = INITIAL_TEMP

    entropies = compute_entropy(current, model, alphabet, bc, device)

    trajectory = [{
        "step": 0, "true_pll": current_pll, "best_pll": best_pll,
        "score": current_score, "best_score": best_score,
        "hamming_to_wt": hamming(current, wt_seq), "temperature": temp,
        "accepted": True, "accepted_downhill": False,
        "pll_delta": 0.0, "eds": current_eds,
    }]

    for step in range(1, MAX_STEPS + 1):
        props = propose_mutations(current, PROPOSALS_PER_STEP, rng, entropies)
        if not props:
            best_score_history.append(best_score)
            steps_since_imp += 1
            trajectory.append({
                "step": step, "true_pll": current_pll, "best_pll": best_pll,
                "score": current_score, "best_score": best_score,
                "hamming_to_wt": hamming(current, wt_seq), "temperature": temp,
                "accepted": False, "accepted_downhill": False,
                "pll_delta": 0.0, "eds": trajectory[-1]["eds"],
            })
            temp = max(temp * COOLING_RATE, MIN_TEMP)
            if steps_since_imp >= PATIENCE: break
            continue

        # MLLR pre-filter
        mllr = [(compute_mllr(current, p, a, model, alphabet, bc, device), p, a, s)
                for p, a, s in props]
        mllr.sort(key=lambda x: x[0], reverse=True)

        # Evaluate top-K with PLL + EDS
        best_cand_score = -float('inf')
        best_cand_seq = None
        best_cand_pll = -float('inf')
        best_cand_eds = current_eds

        for m, pos, aa, seq in mllr[:TOP_K_REEVAL]:
            cpll = score_true_pll(seq, model, alphabet, bc, device)
            cemb = get_embeddings(seq, model, bc, device)
            ceds = compute_eds(cemb, wt_emb)

            # Hard constraint: reject if EDS worsens beyond threshold
            if scoring_mode == "hard":
                if ceds < current_eds - eds_hard_threshold:
                    continue

            cscore = calc_score(cpll, ceds)

            if cscore > best_cand_score:
                best_cand_score = cscore
                best_cand_seq = seq
                best_cand_pll = cpll
                best_cand_eds = ceds

        # MCMC accept
        pll_delta = 0.0; accepted = False; accepted_downhill = False
        if best_cand_seq is not None:
            if mcmc_accept(current_score, best_cand_score, temp, rng):
                pll_delta = best_cand_pll - current_pll
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
            "step": step, "true_pll": current_pll, "best_pll": best_pll,
            "score": current_score, "best_score": best_score,
            "hamming_to_wt": hamming(current, wt_seq), "temperature": temp,
            "accepted": accepted, "accepted_downhill": accepted_downhill,
            "pll_delta": pll_delta, "eds": current_eds,
        })

        temp = max(temp * COOLING_RATE, MIN_TEMP)
        if steps_since_imp >= PATIENCE: break
        if step >= CONVERGENCE_WINDOW:
            if np.var(best_score_history[-CONVERGENCE_WINDOW:]) < VAR_THRESHOLD: break

    n_steps = len(trajectory) - 1
    total_acc = accepted_up + accepted_down
    return trajectory, best_seq, best_pll, {
        "steps": n_steps,
        "accept_rate": total_acc / max(n_steps, 1) * 100,
        "downhill_rate": accepted_down / max(total_acc, 1) * 100,
        "accepted_up": accepted_up, "accepted_down": accepted_down,
        "best_eds": best_eds, "best_score": best_score,
    }


def main():
    t0 = time.perf_counter()
    rng = random.Random(SEED); np.random.seed(SEED)
    outdir = Path(OUTDIR); outdir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta_one(WT_FASTA); L = len(wt_seq)
    model, alphabet, bc = load_esm2(DEVICE)

    wt_pll = score_true_pll(wt_seq, model, alphabet, bc, DEVICE)
    wt_emb = get_embeddings(wt_seq, model, bc, DEVICE)

    total = len(STRATEGIES) * RUNS_PER_STRATEGY
    print(f"WT length: {L} | WT PLL: {wt_pll:.4f}")
    print(f"Strategies: {[s['name'] for s in STRATEGIES]}")
    print(f"Runs per strategy: {RUNS_PER_STRATEGY} | Total: {total}")
    print(f"Hard EDS threshold: {EDS_HARD_THRESHOLD}")
    print(f"MCMC: T0={INITIAL_TEMP}, Tmin={MIN_TEMP}, cool={COOLING_RATE}\n")

    all_results = []
    best_seqs = {}

    start_seqs = [random_sequence(L, rng) for _ in range(RUNS_PER_STRATEGY)]

    for strat in STRATEGIES:
        sname = strat["name"]
        print(f"═══ Strategy: {sname} ═══")

        for i in range(RUNS_PER_STRATEGY):
            start_seq = start_seqs[i]
            start_ham = hamming(start_seq, wt_seq)

            # Each run gets its own rng seeded deterministically
            run_rng = random.Random(SEED + hash(sname) + i)

            traj, best_s, best_p, stats = run_walk(
                start_seq, wt_seq, wt_emb, model, alphabet, bc, DEVICE,
                run_rng, scoring_mode=strat["mode"],
                eds_lambda=strat["lambda"],
                eds_hard_threshold=EDS_HARD_THRESHOLD,
            )

            best_emb = get_embeddings(best_s, model, bc, DEVICE)
            final_eds = compute_eds(best_emb, wt_emb)
            final_ham = hamming(best_s, wt_seq)

            rid = len(all_results)
            all_results.append({
                "run_id": rid, "strategy": sname,
                "color": strat["color"],
                "start_hamming": start_ham,
                "final_pll": best_p, "final_hamming": final_ham,
                "final_eds": final_eds,
                "trajectory": traj, **stats,
            })
            best_seqs[rid] = best_s

            print(f"  Run {i+1}/{RUNS_PER_STRATEGY} | steps={stats['steps']:4d} | "
                  f"PLL={best_p:.4f} | Ham: {start_ham}→{final_ham} | "
                  f"EDS={final_eds:+.1f} | Acc={stats['accept_rate']:.0f}%")
        print()

    rows = []
    for r in all_results:
        for t in r["trajectory"]:
            row = dict(t); row["run_id"] = r["run_id"]
            row["strategy"] = r["strategy"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "trajectory.csv", index=False)

    summary = [{k: v for k, v in r.items() if k != "trajectory"}
               for r in all_results]
    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    for strat in STRATEGIES:
        sr = [r for r in all_results if r["strategy"] == strat["name"]]
        if sr:
            best_r = max(sr, key=lambda r: r["final_pll"])
            safe_name = strat["name"].replace("=", "").replace(".", "")
            (outdir / f"best_{safe_name}.fasta").write_text(
                f">{strat['name']}_pll={best_r['final_pll']:.6f}"
                f"_ham={best_r['final_hamming']}"
                f"_eds={best_r['final_eds']:.1f}\n"
                f"{best_seqs[best_r['run_id']]}\n")

    def get_runs(name):
        return [r for r in all_results if r["strategy"] == name]
    def strat_color(name):
        for s in STRATEGIES:
            if s["name"] == name: return s["color"]
        return "gray"

    snames = [s["name"] for s in STRATEGIES]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scoring Strategy Comparison (all random starts)",
                 fontsize=13, fontweight="bold", y=1.02)

    # (0,0) Best PLL trajectories
    for r in all_results:
        t = r["trajectory"]; c = r["color"]
        axs[0, 0].plot([x["step"] for x in t], [x["best_pll"] for x in t],
                       alpha=0.5, lw=0.8, color=c)
    axs[0, 0].axhline(wt_pll, color="black", ls="--", lw=1, label=f"WT={wt_pll:.3f}")
    legs = [Line2D([0], [0], color=s["color"], lw=2) for s in STRATEGIES]
    legs.append(Line2D([0], [0], color="black", ls="--"))
    axs[0, 0].legend(legs, snames + [f"WT={wt_pll:.3f}"], fontsize=8)
    axs[0, 0].set(title="Best PLL Trajectories", xlabel="Step", ylabel="Best PLL")
    axs[0, 0].grid(True, alpha=0.3)

    # (0,1) Final PLL boxplot by strategy
    pll_data = [[ r["final_pll"] for r in get_runs(s)] for s in snames]
    bp = axs[0, 1].boxplot(pll_data, labels=snames, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(STRATEGIES[i]["color"]); patch.set_alpha(0.7)
    axs[0, 1].axhline(wt_pll, color="black", ls="--", label=f"WT={wt_pll:.3f}")
    axs[0, 1].set(title="Final PLL by Strategy", ylabel="PLL")
    axs[0, 1].legend(fontsize=9); axs[0, 1].grid(True, alpha=0.3, axis="y")

    # (1,0) Final Hamming boxplot by strategy
    ham_data = [[r["final_hamming"] for r in get_runs(s)] for s in snames]
    bp2 = axs[1, 0].boxplot(ham_data, labels=snames, patch_artist=True)
    for i, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(STRATEGIES[i]["color"]); patch.set_alpha(0.7)
    axs[1, 0].set(title="Final Hamming to WT", ylabel="Hamming")
    axs[1, 0].grid(True, alpha=0.3, axis="y")

    # (1,1) Final EDS boxplot by strategy
    eds_data = [[r["final_eds"] for r in get_runs(s)] for s in snames]
    bp3 = axs[1, 1].boxplot(eds_data, labels=snames, patch_artist=True)
    for i, patch in enumerate(bp3["boxes"]):
        patch.set_facecolor(STRATEGIES[i]["color"]); patch.set_alpha(0.7)
    axs[1, 1].axhline(0, color="red", ls="--", label="WT=0")
    axs[1, 1].set(title="Final EDS (closer to 0 = closer to WT)", ylabel="EDS")
    axs[1, 1].legend(fontsize=9); axs[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 1 (comparison): {outdir / 'plots_comparison.png'}")


    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Hamming Distance Analysis",
                   fontsize=13, fontweight="bold", y=1.02)

    # (0,0) Hamming over time
    for r in all_results:
        t = r["trajectory"]; c = r["color"]
        axs2[0, 0].plot([x["step"] for x in t], [x["hamming_to_wt"] for x in t],
                        alpha=0.4, lw=0.8, color=c)
    axs2[0, 0].legend(legs[:len(STRATEGIES)], snames, fontsize=8)
    axs2[0, 0].set(title="Hamming to WT Over Time", xlabel="Step", ylabel="Hamming")
    axs2[0, 0].grid(True, alpha=0.3)

    # (0,1) Hamming change (start - final) by strategy
    change_data = [[r["start_hamming"] - r["final_hamming"] for r in get_runs(s)]
                   for s in snames]
    bp4 = axs2[0, 1].boxplot(change_data, labels=snames, patch_artist=True)
    for i, patch in enumerate(bp4["boxes"]):
        patch.set_facecolor(STRATEGIES[i]["color"]); patch.set_alpha(0.7)
    axs2[0, 1].axhline(0, color="black", lw=0.5)
    axs2[0, 1].set(title="Hamming Change (+ve = toward WT)",
                    ylabel="Start − Final Hamming")
    axs2[0, 1].grid(True, alpha=0.3, axis="y")

    # (1,0) Final PLL vs Final Hamming scatter
    for r in all_results:
        axs2[1, 0].scatter(r["final_hamming"], r["final_pll"],
                            c=r["color"], s=60, edgecolors="black", lw=0.5, alpha=0.8)
    axs2[1, 0].axhline(wt_pll, color="black", ls="--", alpha=0.5)
    axs2[1, 0].legend(legs[:len(STRATEGIES)], snames, fontsize=8)
    axs2[1, 0].set(title="Final PLL vs Hamming", xlabel="Hamming", ylabel="PLL")
    axs2[1, 0].grid(True, alpha=0.3)

    # (1,1) Final EDS vs Final Hamming scatter
    for r in all_results:
        axs2[1, 1].scatter(r["final_hamming"], r["final_eds"],
                            c=r["color"], s=60, edgecolors="black", lw=0.5, alpha=0.8)
    axs2[1, 1].axhline(0, color="red", ls="--", alpha=0.5)
    axs2[1, 1].legend(legs[:len(STRATEGIES)], snames, fontsize=8)
    axs2[1, 1].set(title="Final EDS vs Hamming", xlabel="Hamming", ylabel="EDS")
    axs2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "plots_hamming.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 (hamming):    {outdir / 'plots_hamming.png'}")

    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("MCMC Diagnostics", fontsize=13, fontweight="bold")

    for r in all_results:
        t = r["trajectory"]; c = r["color"]
        axs3[0].plot([x["step"] for x in t], [x["temperature"] for x in t],
                     alpha=0.4, lw=0.8, color=c)
    axs3[0].set_yscale("log"); axs3[0].set(title="Temperature", xlabel="Step")
    axs3[0].grid(True, alpha=0.3)

    for r in all_results:
        t = r["trajectory"]; c = r["color"]
        d = [x["accepted_downhill"] for x in t]
        if len(d) > 1:
            cum = np.cumsum(d) / np.arange(1, len(d)+1) * 100
            axs3[1].plot([x["step"] for x in t], cum, alpha=0.4, lw=0.8, color=c)
    axs3[1].set(title="Cum. Downhill Accept %", xlabel="Step")
    axs3[1].grid(True, alpha=0.3)

    # Acceptance rate boxplot by strategy
    acc_data = [[r["accept_rate"] for r in get_runs(s)] for s in snames]
    bp5 = axs3[2].boxplot(acc_data, labels=snames, patch_artist=True)
    for i, patch in enumerate(bp5["boxes"]):
        patch.set_facecolor(STRATEGIES[i]["color"]); patch.set_alpha(0.7)
    axs3[2].set(title="Final Accept Rate %", ylabel="%")
    axs3[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outdir / "plots_mcmc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 3 (MCMC):       {outdir / 'plots_mcmc.png'}")

    fig4, axs4 = plt.subplots(1, 3, figsize=(18, 5))
    fig4.suptitle("EDS Analysis", fontsize=13, fontweight="bold")

    for r in all_results:
        t = r["trajectory"]; c = r["color"]
        axs4[0].plot([x["step"] for x in t], [x["eds"] for x in t],
                     alpha=0.4, lw=0.8, color=c)
    axs4[0].axhline(0, color="red", ls="--", label="WT=0")
    axs4[0].set(title="EDS Trajectories", xlabel="Step", ylabel="EDS")
    axs4[0].legend(fontsize=9); axs4[0].grid(True, alpha=0.3)

    for r in all_results:
        axs4[1].scatter(r["final_eds"], r["final_pll"],
                        c=r["color"], s=60, edgecolors="black", lw=0.5, alpha=0.8)
    ae = [r["final_eds"] for r in all_results]
    ap = [r["final_pll"] for r in all_results]
    if len(ae) > 2:
        rho = np.corrcoef(ae, ap)[0, 1]
        axs4[1].set_title(f"EDS vs PLL (r={rho:.3f})")
    axs4[1].set(xlabel="Final EDS", ylabel="Final PLL")
    axs4[1].legend(legs[:len(STRATEGIES)], snames, fontsize=8)
    axs4[1].grid(True, alpha=0.3)

    axs4[2].hist([r["final_eds"] for r in all_results], bins=15,
                  edgecolor="black", alpha=0.7, color="#e67e22")
    axs4[2].axvline(0, color="green", ls="--", label="WT=0")
    axs4[2].axvline(np.mean(ae), color="red", ls="--",
                     label=f"Mean={np.mean(ae):.1f}")
    axs4[2].set(title="EDS Distribution", xlabel="EDS", ylabel="Count")
    axs4[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / "plots_eds.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 4 (EDS):        {outdir / 'plots_eds.png'}")

    print(f"\n{'='*70}")
    print(f"WT PLL: {wt_pll:.4f} | WT EDS: 0.00\n")
    print(f"{'Strategy':<12} {'PLL':>8} {'Hamming':>8} {'EDS':>8} {'Steps':>6} {'Acc%':>6}")
    print("-" * 56)
    for sname in snames:
        sr = get_runs(sname)
        pp = np.mean([r["final_pll"] for r in sr])
        hh = np.mean([r["final_hamming"] for r in sr])
        ee = np.mean([r["final_eds"] for r in sr])
        ss = np.mean([r["steps"] for r in sr])
        aa = np.mean([r["accept_rate"] for r in sr])
        print(f"{sname:<12} {pp:>+8.4f} {hh:>8.0f} {ee:>8.1f} {ss:>6.0f} {aa:>5.0f}%")

    print(f"\nRuntime: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"Outputs: {outdir}/")


if __name__ == "__main__":
    main()