#!/bin/bash

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SOTA

# Hyperparameters as variables
ALPHA_TRAIN=0.3               # Weighting of KOT term on frame features in OT during training
ALPHA_EVAL=0.6                # Weighting of KOT term on frame features in OT during evaluation
LAMBDA_FRAMES_TRAIN=0.05      # Penalty for balanced frame assumption during training
LAMBDA_ACTIONS_TRAIN=0.11     # Penalty for balanced action assumption during training
LAMBDA_FRAMES_EVAL=0.05       # Penalty for balanced frame assumption during evaluation
LAMBDA_ACTIONS_EVAL=0.01      # Penalty for balanced action assumption during evaluation
EPS_TRAIN=0.07                # Entropy regularization for OT during training
EPS_EVAL=0.04                 # Entropy regularization for OT during evaluation
RADIUS_GW=0.04                # Radius parameter for GW structure loss
RHO=0.15                      # Global structure weighting factor
N_FRAMES=3                    # Number of frames sampled per video for train/val

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
    --group testing_4_big_e_3 \
    --n-epochs 30 \
    --visualize \
    --n-clusters 3 \
    --val-freq 5 \
    --gpu 0 \
    --seed 0 \
    --batch-size 2 \
    --wandb \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
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
    --n-frames $N_FRAMES
