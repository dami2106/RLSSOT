#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.07668392561
ALPHA_EVAL=0.2906467054

UB_FRAMES=false
UB_ACTIONS=true

LAMBDA_FRAMES_TRAIN=0.03051188463
LAMBDA_ACTIONS_TRAIN=0.05703861099
LAMBDA_FRAMES_EVAL=0.04230932115
LAMBDA_ACTIONS_EVAL=0.09257224927

EPS_TRAIN=0.009768745467
EPS_EVAL=0.0344657881
RADIUS_GW=0.0618802679

# --- Dataset parameters ---
DATASET="stone_pick_random/stone_pick_random_symbolic"
FEATURE_NAME="symbolic_obs"
STD_FEATS=true
SAVE_DIRECTORY="runs"
RUN="stone_pick_random_symbolic"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=7
BATCH_SIZE=16
N_FRAMES=43
LEARNING_RATE=0.00006808331585
WEIGHT_DECAY=0.00000005734383963
LOG=true
VISUALIZE=true
SEED=0
RHO=0.03536435241
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
CMD="$CMD --ub-actions"
CMD="$CMD --std-feats"
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
