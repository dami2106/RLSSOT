#!/bin/bash
#SBATCH --job-name=end_skill_models          # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/pu_end_skill_model_pca750_TUNED.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

python RLSSOT/Skill_Learning/train_end_model_pulearning.py
# python RLSSOT/Skill_Learning/train_end_model.py