#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.6679431214
ALPHA_EVAL=0.9443414454

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.01450610587
LAMBDA_ACTIONS_TRAIN=0.02716573494
LAMBDA_FRAMES_EVAL=0.04172682601
LAMBDA_ACTIONS_EVAL=0.0824780332

EPS_TRAIN=0.1245647852
EPS_EVAL=0.2948249632
RADIUS_GW=0.08046543148

# --- Dataset parameters ---
DATASET="wsws_random/wsws_random_symbolic"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="wsws_random_symbolic"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=11
BATCH_SIZE=2
N_FRAMES=141
LEARNING_RATE=0.0009725110858
WEIGHT_DECAY=2.66E-08
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2309831786
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
