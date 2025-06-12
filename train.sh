#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.4369905979
ALPHA_EVAL=0.6641222536

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.06282911576
LAMBDA_ACTIONS_TRAIN=0.08017023198
LAMBDA_FRAMES_EVAL=0.06461802557
LAMBDA_ACTIONS_EVAL=0.06324252838

EPS_TRAIN=0.004813310568
EPS_EVAL=0.1837366929
RADIUS_GW=0.01891132859

# # Number of OT iterations for training and evaluation && Step Size 
# N_OT_TRAIN="27 1"
# N_OT_EVAL="27 1"
# STEP_SIZE=""

# --- Dataset parameters ---
DATASET="stone_pick_static/stone_pick_static_pixels_big"
FEATURE_NAME="pca_features_with_actions"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_static_pixels_big_actions"
VAL_FREQ=100

# --- General parameters ---
N_EPOCHS=12
BATCH_SIZE=16
N_FRAMES=19
LEARNING_RATE=0.0002476443857
WEIGHT_DECAY=0.0000208
LOG=true
VISUALIZE=true
SEED=0
RHO=0.03433118079
N_CLUSTERS=5
LAYERS="667 300 40"
# USE_KMEANS=true

# --- Build the command ---
CMD="python src/train.py \
  --alpha-train $ALPHA_TRAIN \
  --alpha-eval $ALPHA_EVAL \
  --lambda-frames-train $LAMBDA_FRAMES_TRAIN \
  --lambda-actions-train $LAMBDA_ACTIONS_TRAIN \
  --lambda-frames-eval $LAMBDA_FRAMES_EVAL \
  --lambda-actions-eval $LAMBDA_ACTIONS_EVAL \
  --eps-train $EPS_TRAIN \
  --eps-eval $EPS_EVAL \
  --radius-gw $RADIUS_GW \
  --dataset $DATASET \
  --n-frames $N_FRAMES \
  --save-directory $SAVE_DIRECTORY \
  --n-epochs $N_EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --layers $LAYERS \
  --rho $RHO \
  --n-clusters $N_CLUSTERS \
  --val-freq $VAL_FREQ \
  --seed $SEED \
  --run $RUN \
  --feature-name $FEATURE_NAME"

# Append boolean flags if enabled
if [ "$UB_FRAMES" = true ]; then
  CMD="$CMD --ub-frames"
fi

if [ "$UB_ACTIONS" = true ]; then
  CMD="$CMD --ub-actions"
fi

if [ "$STD_FEATS" = true ]; then
  CMD="$CMD --std-feats"
fi


# # The k-means flag is set to false when provided (i.e., disable initialization with kmeans)
# if [ "$USE_KMEANS" = false ]; then
#   CMD="$CMD --k-means"
# fi

if [ "$VISUALIZE" = true ]; then
  CMD="$CMD --visualize"
fi

if [ "$LOG" = true ]; then
  CMD="$CMD --log"
fi

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD