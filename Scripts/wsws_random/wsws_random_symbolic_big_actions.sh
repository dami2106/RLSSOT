#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.2284158398
ALPHA_EVAL=0.8123260814

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.04551372947
LAMBDA_ACTIONS_TRAIN=0.05767665963
LAMBDA_FRAMES_EVAL=0.009292566326
LAMBDA_ACTIONS_EVAL=0.09499889094

EPS_TRAIN=0.05820562175
EPS_EVAL=0.1993103742
RADIUS_GW=0.07868758996

# --- Dataset parameters ---
DATASET="wsws_random/wsws_random_symbolic_big"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="wsws_random_symbolic_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=6
BATCH_SIZE=16
N_FRAMES=136
LEARNING_RATE=0.01549985372
WEIGHT_DECAY=2.85E-06
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2711001285
N_CLUSTERS=2
LAYERS="1104 512 50"


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
