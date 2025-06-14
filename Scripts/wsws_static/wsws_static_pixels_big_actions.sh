#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.6814967093
ALPHA_EVAL=0.06736574182

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.02403698421
LAMBDA_ACTIONS_TRAIN=0.0282283003
LAMBDA_FRAMES_EVAL=0.03507620648
LAMBDA_ACTIONS_EVAL=0.08951246065

EPS_TRAIN=0.02167192855
EPS_EVAL=0.05883790898
RADIUS_GW=0.004237835809

# --- Dataset parameters ---
DATASET="wsws_static/wsws_static_pixels_big"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="wsws_static_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=15
BATCH_SIZE=2
N_FRAMES=31
LEARNING_RATE=0.02572110032
WEIGHT_DECAY=0.0001237314908
LOG=true
VISUALIZE=true
SEED=0
RHO=0.1771528875
N_CLUSTERS=2
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
CMD="$CMD --ub-frames"
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
