#!/bin/bash

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SOTA

# Hyperparameters as variables
ALPHA_TRAIN=0.4               # Weighting of KOT term on frame features in OT during training
ALPHA_EVAL=0.7                # Weighting of KOT term on frame features in OT during evaluation
LAMBDA_FRAMES_TRAIN=0.05      # Penalty for balanced frame assumption during training
LAMBDA_ACTIONS_TRAIN=0.1   # Penalty for balanced action assumption during training
LAMBDA_FRAMES_EVAL=0.05       # Penalty for balanced frame assumption during evaluation
LAMBDA_ACTIONS_EVAL=0.01      # Penalty for balanced action assumption during evaluation
EPS_TRAIN=0.07                # Entropy regularization for OT during training
EPS_EVAL=0.04                 # Entropy regularization for OT during evaluation
RADIUS_GW=0.04                # Radius parameter for GW structure loss
RHO=0.2                      # Global structure weighting factor
N_FRAMES=3                    # Number of frames sampled per video for train/val

LEARNING_RATE=1e-1            # Learning rate for the optimizer
WEIGHT_DECAY=1e-4             # Weight decay for the optimizer

# Boolean for std-feats
USE_STD_FEATS=true            # Set to true to enable standardization of features

# Translate boolean to flag
if [ "$USE_STD_FEATS" = true ]; then
    STD_FEATS="--std-feats"
else
    STD_FEATS=""
fi

# Training script with specified hyperparameters
python3 train.py \
    --activity all \
    --dataset desktop_assembly \
    --group standard_baseline_random_simple_np\
    --n-epochs 30 \
    --visualize \
    --n-clusters 3 \
    --val-freq 5 \
    --gpu 0 \
    --seed 0 \
    --batch-size 2 \
    --wandb \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --layers 11 11 11 \
    --n-ot-train 25 1 \
    --n-ot-eval 25 1 \
    --alpha-train $ALPHA_TRAIN \
    --alpha-eval $ALPHA_EVAL \
    --lambda-frames-train $LAMBDA_FRAMES_TRAIN \
    --lambda-actions-train $LAMBDA_ACTIONS_TRAIN \
    --lambda-frames-eval $LAMBDA_FRAMES_EVAL \
    --lambda-actions-eval $LAMBDA_ACTIONS_EVAL \
    --eps-train $EPS_TRAIN \
    --eps-eval $EPS_EVAL \
    --radius-gw $RADIUS_GW \
    --rho $RHO \
    $STD_FEATS \
    --n-frames $N_FRAMES\
    -ua \