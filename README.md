# Instructions to run ESMFold on HPC Cluster Curta

## Prerequisites

- HPC account with GPU access
- Python 3.9 available as a module
- Cloned project repository:

```bash
mkdir -p ~/projects
cd ~/projects
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

## Step 2 - Load System Modules

```bash
module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
```

Don't install PyTorch via pip, use this module.

## Step 3 — Create Virtual Environment

```bash
python -m venv venv_esmf
source venv_esmf/bin/activate
```

Check that system Python is not used:

```bash
which python
python -c "import sys; print(sys.prefix)"
```

The expected output should be:

```bash
/home/<user>/projects/esmfold-project/venv_esmf
```

## Step 4 - Install Required Python Packages

```bash
pip install --upgrade pip
```
```bash
pip install \
  omegaconf==2.3.0 \
  einops \
  ml-collections \
  biopython==1.85 \
  dm-tree==0.1.8 \
  biotite
```

And also install ESM from this link and not from pip:

```bash
pip install "fair-esm @ git+https://github.com/facebookresearch/esm.git@main"
```

and OpenFold:

```bash
pip install openfold==1.0.0
```

## Step 5 - Patch OpenFold Utils:

Find OpenFold location:

```bash
python - << 'EOF'
import openfold, os
print(os.path.dirname(openfold.__file__))
EOF
```

Go to utils/ subfolder on that path and replace __init__.py:

```bash
cd <printed-path>/utils
cat > __init__.py << 'EOF'

from . import kernel         
from . import seed           
from . import checkpointing 

__all__ = ["kernel", "seed", "checkpointing"]
EOF
```

Now return to project: 

```bash
cd ~/projects/esmfold-project
```

## Step 6 — Verify Installation

```bash
python - << 'EOF'
import torch, esm, openfold
import torch.cuda
print("torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("esm:", esm.__file__)
print("openfold:", openfold.__file__)
EOF
```

If all paths resolve correctly, the environment is valid.

## Step 7 — Run ESMFold

Run the basics example script:

```bash
python fold_v1.py
```

Expected output:

```bash
Using device: cuda
Mean pLDDT: XX.XX
```

If that works, everything else should work, too. 

# Run predict_cluster.py

This tool folds all FASTA files in a directory using Meta’s ESMFold model and produces:

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

# Run ESM2 Analysis

This script runs ESM2-based analysis on wild-type and mutant protein sequences.

For each sequence, it computes:
- Pseudo Log-Likelihood (PLL)
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
- Mutants must share the same base_id as their WT (es. __insulin_human__ base-id, __insulin_human_mut1_15_A.fasta__ mutant)
- Define a folder (es. __esm2_fasta__) where to put all the FASTAs and a directory for the results (es. __runs__).

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

From the project root

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


















