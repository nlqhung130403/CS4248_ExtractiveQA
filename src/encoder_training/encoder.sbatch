#!/bin/bash
#SBATCH --job-name=squad-encoder
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=t4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%3
#SBATCH --propagate=NONE
#SBATCH --output=/path/to/project/logs/%x_%A_%a.out
#SBATCH --error=/path/to/project/logs/%x_%A_%a.err
set -euo pipefail
ulimit -v unlimited
cd /path/to/project/encoder_training
export TMPDIR=/path/to/scratch/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/project/encoder_deps
PY=/path/to/python
printf 'Start %s host %s array %s task %s\n' "$(date -Is)" "$(hostname)" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
nvidia-smi
"$PY" train_encoder.py
