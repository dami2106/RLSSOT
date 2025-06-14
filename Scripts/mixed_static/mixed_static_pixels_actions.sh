#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.1220949259
ALPHA_EVAL=0.4093377792

UB_FRAMES=true
UB_ACTIONS=true

LAMBDA_FRAMES_TRAIN=0.09464720924
LAMBDA_ACTIONS_TRAIN=0.0436445415
LAMBDA_FRAMES_EVAL=0.07698470605
LAMBDA_ACTIONS_EVAL=0.09996674802

EPS_TRAIN=0.0230401753
EPS_EVAL=0.007829346889
RADIUS_GW=0.08664147941

# --- Dataset parameters ---
DATASET="mixed_static/mixed_static_pixels"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="mixed_static_pixels"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=27
BATCH_SIZE=8
N_FRAMES=20
LEARNING_RATE=0.002685964758
WEIGHT_DECAY=2.32E-07
LOG=true
VISUALIZE=true
SEED=0
RHO=0.1236958918
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
CMD="$CMD --ub-frames"
CMD="$CMD --ub-actions"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
