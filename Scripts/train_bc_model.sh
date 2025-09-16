#!/bin/bash
#SBATCH --job-name=behaviour_cloning        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/behaviour_cloning_pca512_wood.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

python RLSSOT/Skill_Learning/behavioural_cloning.py