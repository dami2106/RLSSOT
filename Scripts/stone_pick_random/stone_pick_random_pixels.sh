#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.1066986524
ALPHA_EVAL=0.3221712105

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.003473626121
LAMBDA_ACTIONS_TRAIN=0.09589952526
LAMBDA_FRAMES_EVAL=0.02880255701
LAMBDA_ACTIONS_EVAL=0.05628788947

EPS_TRAIN=0.02672349459
EPS_EVAL=0.09752644267
RADIUS_GW=0.02189334562

# --- Dataset parameters ---
DATASET="stone_pick_random/stone_pick_random_pixels"
FEATURE_NAME="pca_features"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_random_pixels"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=25
BATCH_SIZE=16
N_FRAMES=27
LEARNING_RATE=0.02435461843
WEIGHT_DECAY=0.0002227395217
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2996569434
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
CMD="$CMD --ub-frames"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
