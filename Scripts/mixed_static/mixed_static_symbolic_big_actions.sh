#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.2261608448
ALPHA_EVAL=0.5023386713

UB_FRAMES=true
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.09655745262
LAMBDA_ACTIONS_TRAIN=0.03583142706
LAMBDA_FRAMES_EVAL=0.09510788763
LAMBDA_ACTIONS_EVAL=0.09802437878

EPS_TRAIN=0.1617841138
EPS_EVAL=0.08705022845
RADIUS_GW=0.05045000262

# --- Dataset parameters ---
DATASET="mixed_static/mixed_static_symbolic_big"
FEATURE_NAME="symbolic_obs_with_actions"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="mixed_static_symbolic_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=20
BATCH_SIZE=8
N_FRAMES=25
LEARNING_RATE=0.01284972178
WEIGHT_DECAY=1.90E-05
LOG=true
VISUALIZE=true
SEED=0
RHO=0.2199381408
N_CLUSTERS=5
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
CMD="$CMD --ub-frames"
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
