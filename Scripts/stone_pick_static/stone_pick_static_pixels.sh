#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.174809031
ALPHA_EVAL=0.1122693538

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.08044277241
LAMBDA_ACTIONS_TRAIN=0.09580566474
LAMBDA_FRAMES_EVAL=0.09695120454
LAMBDA_ACTIONS_EVAL=0.07305196066

EPS_TRAIN=0.05101074488
EPS_EVAL=0.06301592218
RADIUS_GW=0.09637594951

# --- Dataset parameters ---
DATASET="stone_pick_static/stone_pick_static_pixels"
FEATURE_NAME="pca_features"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_static_pixels"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=25
BATCH_SIZE=2
N_FRAMES=85
LEARNING_RATE=0.005017950668
WEIGHT_DECAY=1.03E-08
LOG=true
VISUALIZE=true
SEED=0
RHO=0.09754308224
N_CLUSTERS=5
LAYERS="300 150 40"


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
