#!/bin/bash
#SBATCH --job-name=behaviour_cloning_cnn        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/behaviour_cloning_cnn_wood_tuning_noprune.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

python Skill_Learning/behavioural_cloning_cnn.py --skill wood --study --trials 250
# python RLSSOT/Skill_Learning/behavioural_cloning.py