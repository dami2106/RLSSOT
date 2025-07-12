#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.25
ALPHA_EVAL=0.34

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.01
LAMBDA_ACTIONS_TRAIN=0.05
LAMBDA_FRAMES_EVAL=0.09
LAMBDA_ACTIONS_EVAL=0.07

EPS_TRAIN=0.05
EPS_EVAL=0.013
RADIUS_GW=0.026

# --- Dataset parameters ---
DATASET="cobblestone_finegrained_1_100"
FEATURE_NAME="features"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="cobblestone_finegrained_1_100"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=25
BATCH_SIZE=16
N_FRAMES=505
LEARNING_RATE=0.001
WEIGHT_DECAY=1.0e-08
LOG=true
VISUALIZE=true
SEED=0
RHO=0.061
N_CLUSTERS=29
LAYERS="512 256 40"

# --- Build the command ---
CMD="python src/train.py \
  --alpha-train $ALPHA_TRAIN \
  --alpha-eval $ALPHA_EVAL \
  --lambda-frames-train $LAMBDA_FRAMES_TRAIN \
  --lambda-actions-train $LAMBDA_ACTIONS_TRAIN \
  --lambda-frames-eval $LAMBDA_FRAMES_EVAL \
  --lambda-actions-eval $LAMBDA_ACTIONS_EVAL \
  --eps-train $EPS_TRAIN \
  --eps-eval $EPS_EVAL \
  --radius-gw $RADIUS_GW \
  --dataset $DATASET \
  --n-frames $N_FRAMES \
  --save-directory $SAVE_DIRECTORY \
  --n-epochs $N_EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --layers $LAYERS \
  --rho $RHO \
  --n-clusters $N_CLUSTERS \
  --val-freq $VAL_FREQ \
  --seed $SEED \
  --run $RUN \
  --feature-name $FEATURE_NAME"

# Append boolean flags if enabled
if [ "$UB_FRAMES" = true ]; then
  CMD="$CMD --ub-frames"
fi

if [ "$UB_ACTIONS" = true ]; then
  CMD="$CMD --ub-actions"
fi

if [ "$STD_FEATS" = true ]; then
  CMD="$CMD --std-feats"
fi

if [ "$VISUALIZE" = true ]; then
  CMD="$CMD --visualize"
fi

if [ "$LOG" = true ]; then
  CMD="$CMD --log"
fi

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
