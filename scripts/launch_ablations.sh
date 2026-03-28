#!/bin/bash
set -e

DATASET=${1:?Usage: ./scripts/launch_ablations.sh <dataset_dir> [tag]}
TAG=${2:-""}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASKS=(
    sim2real_image
    sim2real_image_wrist_binary
    sim2real_image_wrist_per_axis_binary
    sim2real_image_wrist_binary_direction
    sim2real_image_tactile_binary
    sim2real_image_tactile_per_axis_binary
    sim2real_image_tactile_binary_direction
)

mkdir -p slurm_logs

echo "Submitting ${#TASKS[@]} ablation jobs..."
echo "Dataset: $DATASET"
echo ""

for task in "${TASKS[@]}"; do
    exp_name="${task}"
    if [ -n "$TAG" ]; then
        exp_name="${task}_${TAG}"
    fi
    job_id=$(sbatch --parsable "$SCRIPT_DIR/sbatch_tillicum.sh" "$task" "$DATASET" "$exp_name")
    echo "  Submitted $exp_name -> job $job_id"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
