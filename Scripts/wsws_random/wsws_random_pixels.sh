#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.7364753391
ALPHA_EVAL=0.1655119286

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.09388373215
LAMBDA_ACTIONS_TRAIN=0.03831867483
LAMBDA_FRAMES_EVAL=0.07992423322
LAMBDA_ACTIONS_EVAL=0.01027865978

EPS_TRAIN=0.1426715575
EPS_EVAL=0.03270318586
RADIUS_GW=0.04274895472

# --- Dataset parameters ---
DATASET="wsws_random/wsws_random_pixels"
FEATURE_NAME="pca_features"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="wsws_random_pixels"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=28
BATCH_SIZE=16
N_FRAMES=73
LEARNING_RATE=0.0001857225175
WEIGHT_DECAY=7.54E-05
LOG=true
VISUALIZE=true
SEED=0
RHO=0.04492604633
N_CLUSTERS=2
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
CMD="$CMD --ub-frames"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
