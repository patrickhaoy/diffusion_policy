#!/bin/bash
#SBATCH --job-name=dp_tactile
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --exclude=g002
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

TASK_CONFIG=${1:?Usage: sbatch sbatch_tillicum.sh <task_config> <dataset_dir> [exp_name]}
DATASET_DIR=${2:?Usage: sbatch sbatch_tillicum.sh <task_config> <dataset_dir> [exp_name]}
EXP_NAME=${3:-$TASK_CONFIG}

mkdir -p slurm_logs

conda activate robodiff

echo "=== Job $SLURM_JOB_ID ==="
echo "Task config: $TASK_CONFIG"
echo "Dataset dir: $DATASET_DIR"
echo "Exp name:    $EXP_NAME"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================="

python train.py \
    --config-name train_mlp_sim2real_image_with_aux_loss_workspace.yaml \
    --config-dir diffusion_policy/config \
    task=$TASK_CONFIG \
    task.dataset.dataset_dir=$DATASET_DIR \
    exp_name=$EXP_NAME
