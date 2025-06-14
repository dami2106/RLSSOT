#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.2971752188
ALPHA_EVAL=0.3019380836

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.09194198657
LAMBDA_ACTIONS_TRAIN=0.05676523923
LAMBDA_FRAMES_EVAL=0.003613372596
LAMBDA_ACTIONS_EVAL=0.04001928544

EPS_TRAIN=0.05251910388
EPS_EVAL=0.0106416738
RADIUS_GW=0.06061495646

# --- Dataset parameters ---
DATASET="stone_pick_static/stone_pick_static_symbolic_big"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="stone_pick_static_symbolic_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=11
BATCH_SIZE=8
N_FRAMES=39
LEARNING_RATE=0.01628079165
WEIGHT_DECAY=0.0006646561728
LOG=true
VISUALIZE=true
SEED=0
RHO=0.291759277
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
