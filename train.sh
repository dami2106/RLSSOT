#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.4
ALPHA_EVAL=0.7

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.05
LAMBDA_ACTIONS_TRAIN=0.05
LAMBDA_FRAMES_EVAL=0.05
LAMBDA_ACTIONS_EVAL=0.01

EPS_TRAIN=0.07
EPS_EVAL=0.04
RADIUS_GW=0.04

# Number of OT iterations for training and evaluation && Step Size 
N_OT_TRAIN="25 1"
N_OT_EVAL="25 1"
STEP_SIZE=""

# --- Dataset  parameters ---
DATASET="wsws_static/wsws_static_symbolic"
FEATURE_NAME="symbolic_obs"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="test_run"
VAL_FREQ=100

# --- General parameters ---
N_EPOCHS=3
BATCH_SIZE=2
N_FRAMES=20
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
LOG=false
VISUALIZE=false
SEED=0
RHO=0.1
N_CLUSTERS=2
LAYERS="1087 500 50"
USE_KMEANS=true



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
  --n-ot-train $N_OT_TRAIN \
  --n-ot-eval $N_OT_EVAL \
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

if [ -n "$STEP_SIZE" ]; then
  CMD="$CMD --step-size $STEP_SIZE"
fi

# The k-means flag is set to false when provided (i.e., disable initialization with kmeans)
if [ "$USE_KMEANS" = false ]; then
  CMD="$CMD --k-means"
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