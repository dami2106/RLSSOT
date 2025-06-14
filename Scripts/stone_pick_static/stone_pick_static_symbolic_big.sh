#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.3033486607
ALPHA_EVAL=0.0308609066

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.08163578758
LAMBDA_ACTIONS_TRAIN=0.02909773039
LAMBDA_FRAMES_EVAL=0.08580398803
LAMBDA_ACTIONS_EVAL=0.02006141174

EPS_TRAIN=0.1535392619
EPS_EVAL=0.04896235773
RADIUS_GW=0.01933355933

# --- Dataset parameters ---
DATASET="stone_pick_static/stone_pick_static_symbolic_big"
FEATURE_NAME="symbolic_obs"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="stone_pick_static_symbolic_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=5
BATCH_SIZE=2
N_FRAMES=64
LEARNING_RATE=0.06848967087
WEIGHT_DECAY=8.77E-06
LOG=true
VISUALIZE=true
SEED=0
RHO=0.03808639058
N_CLUSTERS=5
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
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
