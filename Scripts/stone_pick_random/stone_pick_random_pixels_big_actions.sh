#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.7612452823
ALPHA_EVAL=0.1517514377

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.07785431337
LAMBDA_ACTIONS_TRAIN=0.01316782102
LAMBDA_FRAMES_EVAL=0.0115259902
LAMBDA_ACTIONS_EVAL=0.08900508402

EPS_TRAIN=0.2198038564
EPS_EVAL=0.246675346
RADIUS_GW=0.01287717855

# --- Dataset parameters ---
DATASET="stone_pick_random/stone_pick_random_pixels_big"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_random_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=23
BATCH_SIZE=16
N_FRAMES=85
LEARNING_RATE=0.00004648730181
WEIGHT_DECAY=0.000003790785992
LOG=true
VISUALIZE=true
SEED=0
RHO=0.06885153946
N_CLUSTERS=5
LAYERS="667 300 40"


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
