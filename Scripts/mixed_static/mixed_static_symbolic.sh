#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.1102902048
ALPHA_EVAL=0.5292817367

UB_FRAMES=true
UB_ACTIONS=true

LAMBDA_FRAMES_TRAIN=0.0200063999
LAMBDA_ACTIONS_TRAIN=0.04064136888
LAMBDA_FRAMES_EVAL=0.07421999602
LAMBDA_ACTIONS_EVAL=0.06846782492

EPS_TRAIN=0.1016498979
EPS_EVAL=0.09286277319
RADIUS_GW=0.06022322623

# --- Dataset parameters ---
DATASET="mixed_static/mixed_static_symbolic"
FEATURE_NAME="symbolic_obs"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="mixed_static_symbolic"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=17
BATCH_SIZE=8
N_FRAMES=47
LEARNING_RATE=0.0002069998091
WEIGHT_DECAY=4.19E-06
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2548791061
N_CLUSTERS=5
LAYERS="1087 512 50"


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
