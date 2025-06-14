#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.5616632699
ALPHA_EVAL=0.4673927766

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.08129420709
LAMBDA_ACTIONS_TRAIN=0.0673311296
LAMBDA_FRAMES_EVAL=0.008788361271
LAMBDA_ACTIONS_EVAL=0.007903799889

EPS_TRAIN=0.07694040685
EPS_EVAL=0.03665591492
RADIUS_GW=0.03307116498

# --- Dataset parameters ---
DATASET="wsws_random/wsws_random_pixels"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="wsws_random_pixels"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=9
BATCH_SIZE=2
N_FRAMES=150
LEARNING_RATE=0.01222686265
WEIGHT_DECAY=0.000272323556
LOG=true
VISUALIZE=true
SEED=0
RHO=0.01649054703
N_CLUSTERS=2
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
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
