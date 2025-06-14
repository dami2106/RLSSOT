#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.9454382764
ALPHA_EVAL=0.07085792898

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.09441026599
LAMBDA_ACTIONS_TRAIN=0.08255156552
LAMBDA_FRAMES_EVAL=0.002658151319
LAMBDA_ACTIONS_EVAL=0.04287470717

EPS_TRAIN=0.1439406452
EPS_EVAL=0.01510532927
RADIUS_GW=0.09814926985

# --- Dataset parameters ---
DATASET="wsws_random/wsws_random_pixels_big"
FEATURE_NAME="pca_features"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="wsws_random_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=17
BATCH_SIZE=8
N_FRAMES=106
LEARNING_RATE=2.50E-05
WEIGHT_DECAY=2.88E-05
LOG=true
VISUALIZE=true
SEED=0
RHO=0.1260249008
N_CLUSTERS=2
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
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
