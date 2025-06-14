#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.7626860913
ALPHA_EVAL=0.4188260691

UB_FRAMES=true
UB_ACTIONS=true

LAMBDA_FRAMES_TRAIN=0.05695296303
LAMBDA_ACTIONS_TRAIN=0.03497726368
LAMBDA_FRAMES_EVAL=0.08071699538
LAMBDA_ACTIONS_EVAL=0.09666595157

EPS_TRAIN=0.1539593796
EPS_EVAL=0.02759726178
RADIUS_GW=0.05972066942

# --- Dataset parameters ---
DATASET="mixed_static/mixed_static_symbolic"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="mixed_static_symbolic"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=28
BATCH_SIZE=16
N_FRAMES=145
LEARNING_RATE=4.96E-05
WEIGHT_DECAY=7.75E-08
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2145244507
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
CMD="$CMD --ub-frames"
CMD="$CMD --ub-actions"
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
