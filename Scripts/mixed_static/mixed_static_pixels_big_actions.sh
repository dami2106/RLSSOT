#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.3125096879
ALPHA_EVAL=0.5754815636

UB_FRAMES=true
UB_ACTIONS=true

LAMBDA_FRAMES_TRAIN=0.08303520492
LAMBDA_ACTIONS_TRAIN=0.07946137751
LAMBDA_FRAMES_EVAL=0.03321312776
LAMBDA_ACTIONS_EVAL=0.007227614533

EPS_TRAIN=0.02847343105
EPS_EVAL=0.07314022239
RADIUS_GW=0.0104617818

# --- Dataset parameters ---
DATASET="mixed_static/mixed_static_pixels_big"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="mixed_static_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=20
BATCH_SIZE=8
N_FRAMES=29
LEARNING_RATE=0.000155653385
WEIGHT_DECAY=2.32E-06
LOG=true
VISUALIZE=true
SEED=0
RHO=0.02001552536
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
CMD="$CMD --ub-frames"
CMD="$CMD --ub-actions"
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
