#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.1539734805
ALPHA_EVAL=0.6741038346

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.0474945567
LAMBDA_ACTIONS_TRAIN=0.09400747442
LAMBDA_FRAMES_EVAL=0.02789200719
LAMBDA_ACTIONS_EVAL=0.01547947464

EPS_TRAIN=0.09630981555
EPS_EVAL=0.1775640588
RADIUS_GW=0.05163158385

# --- Dataset parameters ---
DATASET="stone_pick_static/stone_pick_static_symbolic"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="stone_pick_static_symbolic"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=8
BATCH_SIZE=2
N_FRAMES=23
LEARNING_RATE=0.0002701114982
WEIGHT_DECAY=9.13E-08
LOG=true
VISUALIZE=true
SEED=0
RHO=0.09266807186
N_CLUSTERS=5
LAYERS="1104 512 50"


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
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
