#!/bin/bash

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate SOTA

# python3 train.py -ac all -d desktop_assembly --n-epochs 30 --group testing_new_vis --ub-actions --visualize --wandb


# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SOTA

# Training script with specified hyperparameters
python3 train.py \
    --activity all \  # Select all activity classes for training
    --dataset desktop_assembly \  # Dataset to use for training/evaluation
    --group testing_new_vis \  # Group name for experiment tracking in WandB
    --n-epochs 30 \  # Number of epochs for training
    --visualize \  # Enable visualizations during logging
    --n-clusters 3 \  # Number of action clusters
    --val-freq 5 \  # Frequency of validation in epochs
    --gpu 0 \  # GPU ID to use for training
    --seed 0  # Random seed for initialization
    --batch-size 2 \  # Batch size for training
    --wandb \  # Use WandB for logging
    --learning-rate 1e-3 \  # Learning rate for the optimizer
    --weight-decay 1e-4 \  # Weight decay for the optimizer
    --layers 11 128 40 \  # Layer sizes for MLP (input, hidden, output)
    --n-ot-train 25 1 \  # Number of outer and inner iterations for ASOT solver during training
    --n-ot-eval 25 1 \  # Number of outer and inner iterations for ASOT solver during evaluation
    #----------------- Customise Below -----------------#
    --alpha-train 0.3 \  # Weighting of KOT term on frame features in OT during training
    --alpha-eval 0.6 \  # Weighting of KOT term on frame features in OT during evaluation
    --lambda-frames-train 0.05 \  # Penalty for balanced frame assumption during training
    --lambda-actions-train 0.11 \  # Penalty for balanced action assumption during training
    --lambda-frames-eval 0.05 \  # Penalty for balanced frame assumption during evaluation
    --lambda-actions-eval 0.01 \  # Penalty for balanced action assumption during evaluation
    --eps-train 0.07 \  # Entropy regularization for OT during training
    --eps-eval 0.04 \  # Entropy regularization for OT during evaluation
    --radius-gw 0.04 \  # Radius parameter for GW structure loss
    --rho 0.15 \  # Global structure weighting factor
    --std-feats\  # Standardize features per video during preprocessing
    --n-frames 4 # Number of frames sampled per video for train/val