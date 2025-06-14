#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.1516403979
ALPHA_EVAL=0.03066434725

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.06042552074
LAMBDA_ACTIONS_TRAIN=0.01331914194
LAMBDA_FRAMES_EVAL=0.05999336017
LAMBDA_ACTIONS_EVAL=0.0176045789

EPS_TRAIN=0.004741815263
EPS_EVAL=0.1382705562
RADIUS_GW=0.05673055764

# --- Dataset parameters ---
DATASET="stone_pick_static/stone_pick_static_pixels_big"
FEATURE_NAME="pca_features"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_static_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=30
BATCH_SIZE=8
N_FRAMES=36
LEARNING_RATE=0.0003972067616
WEIGHT_DECAY=1.08E-06
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2862943078
N_CLUSTERS=5
LAYERS="650 300 40"


# --- Build the command ---
CMD="python src/train.py \
  --alpha-train $ALPHA_TRAIN \
  --alpha-eval  $ALPHA_EVAL \
  --lambda-frames-train $LAMBDA_FRAMES_TRAIN \
  --lambda-actions-train $LAMBDA_ACTIONS_TRAIN \
  --lambda-frames-eval  $LAMBDA_FRAMES_EVAL \
  --lambda-actions-eval  $LAMBDA_ACTIONS_EVAL \
  --eps-train   $EPS_TRAIN \
  --eps-eval    $EPS_EVAL \
  --radius-gw   $RADIUS_GW \
  --dataset     $DATASET \
  --n-frames    $N_FRAMES \
  --save-directory $SAVE_DIRECTORY \
  --n-epochs    $N_EPOCHS \
  --batch-size  $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --layers      $LAYERS \
  --rho         $RHO \
  --n-clusters  $N_CLUSTERS \
  --val-freq    $VAL_FREQ \
  --seed        $SEED \
  --run         $RUN \
  --feature-name $FEATURE_NAME"
CMD="$CMD --ub-frames"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
