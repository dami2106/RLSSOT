#!/bin/bash
#SBATCH --job-name=behaviour_cloning_cnn        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/behaviour_cloning_cnn_wp.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

python Skill_Learning/behavioural_cloning_cnn.py --skill table
# python RLSSOT/Skill_Learning/behavioural_cloning.py