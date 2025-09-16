#!/bin/bash
#SBATCH --job-name=${FEATURE:-feature_job}
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/dharvey/HiSD/RLSSOT/cluster_logs/%x_%j.log

# --- environment ---
source ~/.bashrc
conda activate SOTA
# export CUBLAS_WORKSPACE_CONFIG=:4096:8   # uncomment if you need this

echo "==[ SLURM INFO ]=="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Feature: ${FEATURE}"
echo "Layers:  ${LAYERS}"
echo "Filename:${FILENAME}"
echo "Dataset: ${DATASET}"

# --- run ---
python Helpers/optuna_study.py \
  --dataset="${DATASET}" \
  --feature-name="${FEATURE}" \
  --clusters=5 \
  --layers="${LAYERS}" \
  --trials=1500 \
  --filename="${FILENAME}"