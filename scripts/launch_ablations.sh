#!/bin/bash
set -e

DATASET=${1:?Usage: ./scripts/launch_ablations.sh <dataset_dir>}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASKS=(
    sim2real_image
    sim2real_image_wrist_force
    sim2real_image_wrist_wrench
    sim2real_image_finger_ly
    sim2real_image_finger_full
    sim2real_image_finger_friction
)

mkdir -p slurm_logs

echo "Submitting ${#TASKS[@]} ablation jobs..."
echo "Dataset: $DATASET"
echo ""

for task in "${TASKS[@]}"; do
    job_id=$(sbatch --parsable "$SCRIPT_DIR/sbatch_tillicum.sh" "$task" "$DATASET" "$task")
    echo "  Submitted $task -> job $job_id"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
