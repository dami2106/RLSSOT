#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.9959952033
ALPHA_EVAL=0.4285263776

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.05276513904
LAMBDA_ACTIONS_TRAIN=0.0077495354
LAMBDA_FRAMES_EVAL=0.09692044611
LAMBDA_ACTIONS_EVAL=0.03801857102

EPS_TRAIN=0.1442967197
EPS_EVAL=0.1076448409
RADIUS_GW=0.04891980202

# --- Dataset parameters ---
DATASET="wsws_static/wsws_static_symbolic_big"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="wsws_static_symbolic_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=25
BATCH_SIZE=2
N_FRAMES=59
LEARNING_RATE=2.09E-05
WEIGHT_DECAY=0.0004855215735
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2600035948
N_CLUSTERS=2
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
