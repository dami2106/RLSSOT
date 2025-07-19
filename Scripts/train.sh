#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.6
ALPHA_EVAL=0.26

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.06
LAMBDA_ACTIONS_TRAIN=0.09
LAMBDA_FRAMES_EVAL=0.08
LAMBDA_ACTIONS_EVAL=0.09

EPS_TRAIN=0.015
EPS_EVAL=0.027
RADIUS_GW=0.007

# --- Dataset parameters ---
DATASET="wsws_static/wsws_static_pixels_big"
FEATURE_NAME="pca_features"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="wsws_static_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=50
BATCH_SIZE=16
N_FRAMES=20
LEARNING_RATE=0.1
WEIGHT_DECAY=0.001
LOG=true
VISUALIZE=true
SEED=0
RHO=0.113
N_CLUSTERS=2
LAYERS="650 300 40"

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
