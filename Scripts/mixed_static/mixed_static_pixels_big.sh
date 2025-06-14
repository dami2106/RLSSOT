#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.07434594466
ALPHA_EVAL=0.8230799961

UB_FRAMES=true
UB_ACTIONS=true

LAMBDA_FRAMES_TRAIN=0.08423682501
LAMBDA_ACTIONS_TRAIN=0.0825285425
LAMBDA_FRAMES_EVAL=0.04217558672
LAMBDA_ACTIONS_EVAL=0.03039409858

EPS_TRAIN=0.005432281575
EPS_EVAL=0.01340313487
RADIUS_GW=0.01458080174

# --- Dataset parameters ---
DATASET="mixed_static/mixed_static_pixels_big"
FEATURE_NAME="pca_features"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="mixed_static_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=18
BATCH_SIZE=16
N_FRAMES=36
LEARNING_RATE=0.0001105710232
WEIGHT_DECAY=0.0001344258452
LOG=true
VISUALIZE=true
SEED=0
RHO=0.02494160656
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
CMD="$CMD --ub-actions"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
