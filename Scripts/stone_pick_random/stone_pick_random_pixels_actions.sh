#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.9479653362
ALPHA_EVAL=0.5759840102

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.06512533326
LAMBDA_ACTIONS_TRAIN=0.09913215139
LAMBDA_FRAMES_EVAL=0.08255989981
LAMBDA_ACTIONS_EVAL=0.0880712795

EPS_TRAIN=0.2320950278
EPS_EVAL=0.1288094818
RADIUS_GW=0.01158490212

# --- Dataset parameters ---
DATASET="stone_pick_random/stone_pick_random_pixels"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_random_pixels"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=25
BATCH_SIZE=2
N_FRAMES=126
LEARNING_RATE=0.00005253947787
WEIGHT_DECAY=0.00002659606824
LOG=true
VISUALIZE=true
SEED=0
RHO=0.263612048
N_CLUSTERS=5
LAYERS="317 150 40"


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
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
