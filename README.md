# Instructions to run ESMFold on HPC Cluster Curta

## Prerequisites

- HPC account with GPU access
- Access to scratch filesystem at `/scratch/<user>/`
- Cloned project repository:

```bash
cd /scratch/sergeea99/Protein_space
git clone https://github.com/silviacambiago/esmfold-project.git
cd esmfold-project
```

## Step 1 — Start GPU Session

```bash
salloc \
  --partition=gpu \
  --qos=standard \
  --account=agbelik \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=08:00:00
```

A GPU node (g005, g006, ...) should be available now.

## Step 2 — Load System Modules

```bash
module purge
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1
```

> **Note:** Do NOT load the `PyTorch/1.10.0-foss-2021a-CUDA-11.3.1` bundle module.
> Load Python and CUDA separately, then install PyTorch via pip (Step 5).

## Step 3 — Create Clean Virtual Environment

```bash
cd /scratch/sergeea99/Protein_space
python -m venv venv_esmf_clean
source venv_esmf_clean/bin/activate
```

The venv must be created without `--system-site-packages` to avoid conflicts.

Verify:

```bash
which python
python -c "import sys; print(sys.prefix)"
```

## Step 4 — Upgrade pip

```bash
pip install --upgrade pip
```

## Step 5 — Install PyTorch 1.10.2 with CUDA 11.3

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

## Step 6 — Install Compatible NumPy (CRITICAL)

```bash
pip install "numpy<1.24"
```

> **Warning:** NumPy >= 1.24 breaks ESM and OpenFold imports. This pin is required.
> Install this BEFORE other packages so nothing upgrades it.

## Step 7 — Install Core Dependencies

```bash
pip install wheel ninja
pip install \
  omegaconf==2.3.0 \
  einops \
  ml-collections \
  biopython==1.85 \
  dm-tree==0.1.8 \
  biotite \
  modelcif \
  pandas \
  matplotlib
pip install "fair-esm @ git+https://github.com/facebookresearch/esm.git@main"
pip install git+https://github.com/NVIDIA/dllogger.git
```

## Step 8 — Install PyTorch Lightning Stack (Compatible Versions)

```bash
pip install "pytorch-lightning==1.4.9"
pip install --no-deps "torchmetrics==0.6.0"
```

## Step 9 — Install OpenFold v1.0.0 Without Dependencies

```bash
pip install --no-deps --no-build-isolation \
  "openfold @ git+https://github.com/aqlaboratory/openfold.git@v1.0.0"
```

> **Note:** Use `--no-deps` to prevent OpenFold from upgrading PyTorch or NumPy.
> Do NOT use `pip install openfold==1.0.0` (PyPI version pulls incompatible dependencies).
> No manual patching of `openfold/utils/__init__.py` is needed with this install method.

## Step 10 — Force-Reinstall PyTorch (Protective Step)

```bash
pip install --force-reinstall --no-deps \
  torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

This guards against any upstream package silently upgrading PyTorch.

## Step 11 — Verify Installation

```bash
python - << 'EOF'
import torch, esm, openfold
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("esm:", esm.__file__)
print("openfold:", openfold.__file__)
EOF
```

## Step 12 — Test ESMFold

```bash
cd esmfold-project
python fold_v1.py
```

Expected output:

```
Using device: cuda
Mean pLDDT: XX.XX
```

If that works, everything else should work, too.

---

# Run predict_cluster.py

This tool folds all FASTA files in a directory using Meta's ESMFold model and produces:

- One PDB structure per sequence
- A summary.tsv file including sequence length and mean pLDDT confidence score

## Prepare input FASTA files

Place one or more protein sequences in a directory __fasta_files__. Each file must contain exactly one sequence.

## Run folding

```bash
python predict_cluster.py \
    --fasta-dir fasta_files \
    --outdir runs/my_protein \
    --device cuda \
    --chunk-size 128
```

After execution, the folder looks like:

```bash
runs/my_protein/
    protein1.pdb
    protein2.pdb
    ...
    summary.tsv
```

---

# Run ESM2 Analysis

This script runs ESM2-based analysis on wild-type and mutant protein sequences.

For each sequence, it computes:
- Pseudo Log-Likelihood (PLL): sum_i log p(x_i | x_{-i}) via masking
- Average log-probability per residue (avg_log_prob / avPLL)
- Pseudo-perplexity
- Log-Likelihood Ratio vs WT (LLR) for mutated positions only
- Number and positions of mutations

All results are written to a single CSV file:

```bash
esm2_scores.csv
```

## FASTA naming conventions

- All FASTA files must contain exactly one sequence.
- WTs must end with ___wt.fasta__.
- Mutants must share the same base_id as their WT (e.g. __insulin_human__ base-id, __insulin_human_mut1_15_A.fasta__ mutant)
- Define a folder (e.g. __esm2_fasta__) for FASTAs and a directory for results (e.g. __runs__).

Recommended directory structure:

```bash
esmfold-project/
│
├── esm2_fasta/
│   ├── insulin_human_wt.fasta
│   ├── insulin_human_mut1_...
│   ├── insulin_bovin_wt.fasta
│   └── ...
│
├── esm2_analysis_cluster.py
└── runs/
```

## Run the script

From the project root:

```bash
python score_cluster.py \
    --fasta-dir esm2_fasta \
    --outdir runs/esm2_insulin \
    --model esm2_t33_650M_UR50D \
    --device cuda
```

For a more accurate but slower model:

```bash
--model esm2_t36_3B_UR50D
```

After completion, the output directory contains:

```bash
runs/esm2_insulin/
└── esm2_scores.csv
```

## Output column description

| Column | Description |
|------|-------------|
| `base_id` | Protein family / reference identifier shared by WT and mutants |
| `variant_id` | Unique sequence ID (from FASTA filename) |
| `sequence` | Amino acid sequence |
| `is_wt` | `1` = wild type, `0` = mutant |
| `length` | Sequence length (amino acids) |
| `log_pseudo_likelihood` | Total pseudo log-likelihood under ESM2 |
| `avg_log_prob` | Mean log-probability per residue (avPLL) |
| `pseudo_perplexity` | `exp(-avg_log_prob)` |
| `n_mut_positions` | Number of mutations vs WT |
| `llr_vs_wt` | Log-likelihood ratio vs WT at mutated sites |
| `mut_positions` | Mutated positions (0-based, `;`-separated) |

---

# Plot ESM2 Score Distributions

The script __plot_distribution_cluster.py__ reads __esm2_scores.csv__ and:

- Groups sequences by mutation step (__n_mut_positions__)
- Plots one histogram per step marking the WT, too, plus one general
- Saves all plots as PNG files

## Run the script

From the project root:

```bash
python plot_distribution_cluster.py \
    --input runs/esm2_insulin/esm2_scores.csv \
    --outdir runs/esm2_insulin/plots \
    --metric avg_log_prob
```

In general, __--input__ is the path to __esm2_scores.csv__, __--outdir__ is the directory where plots will be saved and __--metric__ is the metric to plot.

After execution:

```bash
runs/esm2_insulin/
├── esm2_scores.csv
└── plots/
    ├── avg_log_prob_ALL.png
    ├── avg_log_prob_step0.png
    ├── avg_log_prob_step1.png
    ├── avg_log_prob_step2.png
    ├── avg_log_prob_step3.png
    └── ...
```

---

# Evolution Walk (ESM2-guided sequence optimisation)

The evolution walk performs a greedy hill-climb in ESM2 score space.
Starting from a random sequence of the same length as the wild type, it
iteratively proposes single-point mutations and accepts any move that does not
decrease the ESM2 pseudo-log-likelihood score. The walk halts automatically
when the score has converged.

## Script: `evolution_game_convergence.py`

This is the current main runner. It loops over **all** wild-type FASTAs in
`wild_types/` in a single invocation, running one independent walk per protein.

### Algorithm

1. **Initialisation** — a fully random amino-acid sequence of length L is
   generated (seeded by `--seed-init`). The starting point is therefore
   completely unrelated to the wild type.
2. **Proposal** — at each step, `PROPOSALS_PER_STEP` (default 100) candidate
   single-point mutations are drawn. Positions and amino acids are chosen
   uniformly at random (seeded by `--seed-mutate`).
3. **Selection** — all candidates are scored in a single GPU batch. The
   highest-scoring candidate is accepted if its score ≥ the current score
   (`non_negative` move rule); otherwise the walk stays.
4. **Convergence check** — after `MIN_STEPS` (1 000) the walk stops when
   either of two criteria is met:
   - No improvement to the best score for `PATIENCE_STEPS` (1 000) consecutive
     steps, **or**
   - Rolling score variance over the last 100 steps falls below `1e-10`.
   A hard ceiling of `MAX_STEPS` (50 000) acts as a safety limit.

Two independent random seeds are used:

| Seed | Controls |
|---|---|
| `--seed-init` | What the starting sequence is |
| `--seed-mutate` | Which mutations are proposed at every step |

Separating them means you can fix the mutation process and vary only the
starting point, or vice versa.

### Run

```bash
python evolution_game_convergence.py \
    --wt-dir wild_types \
    --outdir evolution_walk_out_total_random \
    --seed-init 42 \
    --seed-mutate 1000
```

All arguments are optional; the defaults above are used if omitted.

### Output

One subdirectory is created per protein under `--outdir`:

```
evolution_walk_out_total_random/
├── insulin_human_wt/
│   ├── trajectory.csv        — per-step: score, best_score_so_far, hamming_to_wt, sequence
│   ├── best.fasta            — highest-scoring sequence found
│   ├── convergence_info.txt  — stop reason, best score, step count
│   └── score_curve.png       — three-panel plot: score, Hamming distance, rolling variance
├── ACTB/
│   └── ...
└── ...
```

#### `trajectory.csv` columns

| Column | Description |
|---|---|
| `step` | Walk step index |
| `score` | ESM2 score of current sequence at this step |
| `best_score_so_far` | Running maximum score |
| `hamming_to_wt` | Number of positions differing from the wild type |
| `pos` | Position mutated at this step |
| `aa` | Amino acid introduced at this step |
| `accepted` | `1` if the move was accepted, `0` otherwise |
| `sequence` | Full amino-acid sequence at this step |

### Running multiple seeds on HPC — `run_convergence_all_wt.slurm`

The preferred way to run many chains on Curta is the SLURM array job.
It submits 90 independent chains for one protein in a single `sbatch` call.
Each array task gets a unique seed pair derived from the base seeds.

**Single protein:**
```bash
sbatch --export=PROTEIN_IDX=0,SEED_INIT=42,SEED_MUTATE=1000 \
    run_convergence_all_wt.slurm
```

**All proteins (loop):**
```bash
for i in {0..10}; do
    sbatch --export=PROTEIN_IDX=$i,SEED_INIT=42,SEED_MUTATE=1000 \
        run_convergence_all_wt.slurm
done
```

**Parameters passed via `--export`:**

| Parameter | Default | Description |
|---|---|---|
| `PROTEIN_IDX` | `0` | Index into sorted `wild_types/` file list |
| `SEED_INIT` | `42` | Base seed for starting sequence; task gets `SEED_INIT + task_id` |
| `SEED_MUTATE` | `1000` | Base seed for mutation proposals; task gets `SEED_MUTATE + task_id` |

**To find which index corresponds to which protein:**
```bash
ls -1 wild_types/*.fasta.gz wild_types/*.fasta 2>/dev/null | sort | cat -n
```

**Output structure:**
```
evolution_walk_out_total_random/
└── ACTB/
    ├── run_000_seed42/
    │   ├── trajectory.csv
    │   ├── best.fasta
    │   ├── convergence_info.txt
    │   └── score_curve.png
    ├── run_001_seed43/
    └── ...
```

> **QOS limit:** The cluster enforces a per-user job submit limit. Each
> array job (`--array=0-89`) counts as 90 jobs. If you hit
> `QOSMaxSubmitJobPerUserLimit`, either wait for running jobs to clear or
> reduce the array size at submission: `sbatch --array=0-29 --export=...`

---

## Script: `analyze_batch_results.py`

Aggregates `trajectory.csv` files from **multiple seeds of a single protein**
into summary statistics and comparison plots.

> **Scope:** this script is designed for one protein, many seeds. It has no
> awareness of protein identity and assumes all runs have the same sequence
> length and the same number of steps. Passing runs from different proteins
> will produce meaningless plots and will crash on the mean/std calculation
> (ragged trajectory arrays).

The expected directory layout is the output of running
`evolution_game_convergence.py` repeatedly with different `--seed-init` values
into numbered subdirectories:

```
evolution_convergence_results/
├── run_00_seed42/
│   └── trajectory.csv
├── run_01_seed43/
│   └── trajectory.csv
└── ...
```

### Run

```bash
python analyze_batch_results.py \
    --batch-dir evolution_convergence_results
```

### Output

Written into `--batch-dir`:

| File | Description |
|---|---|
| `batch_summary.csv` | One row per run: best score, final Hamming distance, steps, final sequence |
| `best_overall.fasta` | Highest-scoring sequence across all seeds |
| `batch_analysis.png` | 2×2 panel: trajectories, score distribution, Hamming over time, score vs Hamming |
| `all_score_trajectories.png` | Current and best-score trajectories for all seeds |
| `all_hamming_trajectories.png` | Hamming-to-WT trajectories with mean ± std across seeds |
| `score_evolution_stats.png` | Best-score trajectories with mean ± std band across seeds |

Console output prints per-seed best scores and a summary table (max, mean,
median, std, min) across all seeds.

---

# Diverged-Start Evolution Walk

## Script: `evolution_game_diverged_start.py`

A variant of `evolution_game_convergence.py` where the starting sequence
is not fully random — it is the wild type with a controlled fraction of
positions mutated.

This lets you test whether the walk converges to the same fitness attractor
regardless of starting distance from WT, and at what divergence level the
walk can no longer recover WT-like scores.

### Starting sequence construction

The `--start-divergence` parameter (0.0–1.0) controls initialisation:

| `--start-divergence` | Starting point |
|---|---|
| `0.0` | Wild-type sequence exactly |
| `0.2` | WT with 20% of positions randomly mutated |
| `0.5` | WT with 50% of positions randomly mutated |
| `1.0` | Fully random sequence (same as `evolution_game_convergence.py`) |

Exactly `round(L × divergence)` positions are selected at random
(controlled by `--seed-init`) and each is replaced with a uniformly
random amino acid that is different from the WT residue at that position.
Everything else — the walk, convergence criteria, outputs — is identical
to `evolution_game_convergence.py`.

### Run

```bash
python evolution_game_diverged_start.py \
    --wt-dir           wild_types \
    --protein          insulin_human_wt \
    --outdir           evolution_walk_out_diverged/insulin_human_wt/div020pct/run_000_seed42 \
    --start-divergence 0.2 \
    --seed-init        42 \
    --seed-mutate      1000
```

Omit `--protein` to run all proteins sequentially. Omit `--start-divergence`
to default to `0.0` (WT start).

### Output

Same structure as `evolution_game_convergence.py`, with two additions in
`convergence_info.txt`:

```
Start divergence: 0.2000
Start Hamming to WT: 15
```

The `score_curve.png` Hamming panel also shows a horizontal reference line
at the starting Hamming distance.

---

## Script: `run_evolution_diverged_start.slurm`

SLURM array job for `evolution_game_diverged_start.py`.
One submission per protein per divergence level, 90 chains per submission.
All parameters are passed via `--export`.

**Parameters passed via `--export`:**

| Parameter | Default | Description |
|---|---|---|
| `PROTEIN_IDX` | `0` | Index into sorted `wild_types/` file list |
| `DIVERGENCE` | `0.2` | Starting distance from WT (0.0 = WT, 1.0 = fully random) |
| `SEED_INIT` | `42` | Base seed for starting sequence; task gets `SEED_INIT + task_id` |
| `SEED_MUTATE` | `1000` | Base seed for mutation proposals; task gets `SEED_MUTATE + task_id` |

### Single protein, one divergence level

```bash
sbatch --export=PROTEIN_IDX=0,DIVERGENCE=0.2,SEED_INIT=42,SEED_MUTATE=1000 \
    run_evolution_diverged_start.slurm
```

### All proteins, one divergence level

```bash
for i in {0..10}; do
    sbatch --export=PROTEIN_IDX=$i,DIVERGENCE=0.2,SEED_INIT=42,SEED_MUTATE=1000 \
        run_evolution_diverged_start.slurm
done
```

### Sweep divergence levels for one protein

```bash
for div in 0.0 0.1 0.2 0.3 0.5 1.0; do
    sbatch --export=PROTEIN_IDX=0,DIVERGENCE=$div,SEED_INIT=42,SEED_MUTATE=1000 \
        run_evolution_diverged_start.slurm
done
```

> **QOS limit:** Each array job (`--array=0-89`) counts as 90 jobs toward
> the per-user submit limit. A divergence sweep of 6 levels = 540 jobs,
> which will typically exceed the quota. Submit one level at a time and
> wait for each batch to clear, or reduce array size:
> `sbatch --array=0-29 --export=...` (30 chains per level).

### Output structure

```
evolution_walk_out_diverged/
└── insulin_human_wt/
    ├── div000pct/          ← start from WT
    │   ├── run_000_seed42/
    │   │   ├── trajectory.csv
    │   │   ├── best.fasta
    │   │   ├── convergence_info.txt
    │   │   └── score_curve.png
    │   └── run_001_seed43/
    ├── div020pct/
    ├── div050pct/
    └── div100pct/          ← fully random start
```

Results from a divergence sweep can be analysed with
`analyze_batch_results.py` pointed at a single divergence level directory,
or compared across levels manually via `summary.csv`.
