#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.9237159548
ALPHA_EVAL=0.9025090024

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.01801636946
LAMBDA_ACTIONS_TRAIN=0.02690486685
LAMBDA_FRAMES_EVAL=0.006919545113
LAMBDA_ACTIONS_EVAL=0.02355696592

EPS_TRAIN=0.06056375528
EPS_EVAL=0.2916239587
RADIUS_GW=0.09063594657

# --- Dataset parameters ---
DATASET="wsws_random/wsws_random_symbolic"
FEATURE_NAME="symbolic_obs"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="wsws_random_symbolic"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=25
BATCH_SIZE=8
N_FRAMES=119
LEARNING_RATE=0.001117079078
WEIGHT_DECAY=1.33E-06
LOG=true
VISUALIZE=true
SEED=0
RHO=0.03794636316
N_CLUSTERS=2
LAYERS="1087 512 50"


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
