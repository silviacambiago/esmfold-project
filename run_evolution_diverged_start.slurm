#!/bin/bash
#SBATCH --job-name=evo_div_
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=agbelik
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/evo_div_%A_%a.log
#SBATCH --error=logs/evo_div_%A_%a.err
#SBATCH --exclude=g001,g003,g005,g007

# SLURM Job Array — diverged-start evolution walk, one job per protein.
# PROTEIN_IDX selects the protein (0-10, sorted alphabetically from wild_types/).
# DIVERGENCE sets the starting distance from WT (0.0–1.0).
# SLURM_ARRAY_TASK_ID is the chain index (0-89).
#
# Single protein, one divergence level:
#   sbatch --export=PROTEIN_IDX=0,DIVERGENCE=0.2 run_evolution_diverged_start.slurm
#
# All proteins, one divergence level:
#   for i in {0..10}; do
#       sbatch --export=PROTEIN_IDX=$i,DIVERGENCE=0.2 run_evolution_diverged_start.slurm
#   done
#
# Sweep divergence levels for one protein:
#   for div in 0.0 0.1 0.2 0.3 0.5 1.0; do
#       sbatch --export=PROTEIN_IDX=0,DIVERGENCE=$div run_evolution_diverged_start.slurm
#   done

echo "========================================================================"
echo "  DIVERGED-START EVOLUTION WALK (array job)"
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

# Parameters — provided via --export at submission, with defaults
PROTEIN_IDX=${PROTEIN_IDX:-0}
DIVERGENCE=${DIVERGENCE:-0.2}
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

SEED_INIT=$(( ${SEED_INIT:-42} + SLURM_ARRAY_TASK_ID ))
SEED_MUTATE=$(( ${SEED_MUTATE:-1000} + SLURM_ARRAY_TASK_ID ))

# Format divergence as integer percent for directory name (0.2 → div20pct)
DIV_PCT=$(python3 -c "print(f'div{int(float(\"$DIVERGENCE\")*100):03d}pct')")
OUTDIR="evolution_walk_out_diverged/${PROTEIN}/${DIV_PCT}/run_$(printf "%03d" $CHAIN_IDX)_seed${SEED_INIT}"

echo "This task:"
echo "  Task:        $SLURM_ARRAY_TASK_ID"
echo "  Protein:     $PROTEIN  (index $PROTEIN_IDX)"
echo "  Chain:       $CHAIN_IDX / 89"
echo "  Divergence:  $DIVERGENCE  ($DIV_PCT)"
echo "  Seed init:   $SEED_INIT"
echo "  Seed mutate: $SEED_MUTATE"
echo "  Output:      $OUTDIR"
echo ""

echo "========================================================================"
echo "  RUNNING"
echo "========================================================================"
echo ""

python3 evolution_game_diverged_start.py \
    --wt-dir          wild_types \
    --protein         "$PROTEIN" \
    --outdir          "$OUTDIR" \
    --start-divergence "$DIVERGENCE" \
    --seed-init       "$SEED_INIT" \
    --seed-mutate     "$SEED_MUTATE"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Task $SLURM_ARRAY_TASK_ID ($PROTEIN, div=$DIVERGENCE, chain $CHAIN_IDX) done"
echo "  Exit code: $EXIT_CODE"
echo "  End: $(date)"

if [ -f "$OUTDIR/convergence_info.txt" ]; then
    echo ""
    echo "  Convergence info:"
    cat "$OUTDIR/convergence_info.txt" | sed 's/^/    /'
fi
echo "========================================================================"

exit $EXIT_CODE
