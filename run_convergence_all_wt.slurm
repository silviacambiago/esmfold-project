#!/bin/bash
#SBATCH --job-name=evo_all_wt
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=agbelik
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/evo_all_wt_%A_%a.log
#SBATCH --error=logs/evo_all_wt_%A_%a.err
#SBATCH --exclude=g001,g003,g005,g007

# SLURM Job Array — 90 chains per protein, submitted once per protein.
# PROTEIN_IDX selects the protein (0-10, sorted alphabetically from wild_types/).
# SLURM_ARRAY_TASK_ID is the chain index (0-89).
#
# Single protein:
#   sbatch --export=PROTEIN_IDX=0,SEED_INIT=42,SEED_MUTATE=1000 run_convergence_all_wt.slurm
#
# All proteins:
#   for i in {0..10}; do
#       sbatch --export=PROTEIN_IDX=$i,SEED_INIT=42,SEED_MUTATE=1000 run_convergence_all_wt.slurm
#   done

echo "========================================================================"
echo "  CONVERGENCE-BASED EVOLUTION — ALL WILD TYPES (array job)"
echo "========================================================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start:        $(date)"
echo "========================================================================"
echo ""

module purge
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

cd /scratch/sergeea99/Protein_space/esmfold-project || exit 1
source /scratch/sergeea99/Protein_space/venv_esmf_clean/bin/activate

mkdir -p logs

DELAY=$((RANDOM % 30))
echo "Startup delay: ${DELAY}s (prevents CUDA race)"
sleep $DELAY

echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# PROTEIN_IDX is provided at submission via --export (default 0).
# SLURM_ARRAY_TASK_ID is the chain index within that protein (0-89).
PROTEIN_IDX=${PROTEIN_IDX:-0}
CHAIN_IDX=$SLURM_ARRAY_TASK_ID

mapfile -t WT_FILES < <(ls -1 wild_types/*.fasta.gz wild_types/*.fasta 2>/dev/null | sort)
N_WT=${#WT_FILES[@]}

if [ "$PROTEIN_IDX" -ge "$N_WT" ]; then
    echo "ERROR: PROTEIN_IDX $PROTEIN_IDX >= number of WT files ($N_WT). Nothing to do."
    exit 0
fi

WT_FILE="${WT_FILES[$PROTEIN_IDX]}"
PROTEIN=$(basename "$WT_FILE" .fasta.gz)
PROTEIN=$(basename "$PROTEIN" .fasta)

# Each chain gets a unique seed pair. Base seeds default to 42/1000.
SEED_INIT=$(( ${SEED_INIT:-42} + SLURM_ARRAY_TASK_ID ))
SEED_MUTATE=$(( ${SEED_MUTATE:-1000} + SLURM_ARRAY_TASK_ID ))

OUTDIR="evolution_walk_out_total_random/${PROTEIN}/run_$(printf "%03d" $CHAIN_IDX)_seed${SEED_INIT}"

echo "This task:"
echo "  Task:        $SLURM_ARRAY_TASK_ID"
echo "  Protein:     $PROTEIN  (index $PROTEIN_IDX)"
echo "  Chain:       $CHAIN_IDX / 89"
echo "  Seed init:   $SEED_INIT"
echo "  Seed mutate: $SEED_MUTATE"
echo "  Output:      $OUTDIR"
echo ""

echo "========================================================================"
echo "  RUNNING"
echo "========================================================================"
echo ""

python3 evolution_game_convergence.py \
    --wt-dir      wild_types \
    --protein     "$PROTEIN" \
    --outdir      "$OUTDIR" \
    --seed-init   "$SEED_INIT" \
    --seed-mutate "$SEED_MUTATE"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Task $SLURM_ARRAY_TASK_ID ($PROTEIN chain $CHAIN_IDX) done"
echo "  Exit code: $EXIT_CODE"
echo "  End: $(date)"

if [ -f "$OUTDIR/convergence_info.txt" ]; then
    echo ""
    echo "  Convergence info:"
    cat "$OUTDIR/convergence_info.txt" | sed 's/^/    /'
fi
echo "========================================================================"

exit $EXIT_CODE
