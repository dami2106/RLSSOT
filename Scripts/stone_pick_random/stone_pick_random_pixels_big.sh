#!/bin/bash

# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN=0.5449613686
ALPHA_EVAL=0.5772286787

UB_FRAMES=false
UB_ACTIONS=false

LAMBDA_FRAMES_TRAIN=0.0244374232
LAMBDA_ACTIONS_TRAIN=0.0873559363
LAMBDA_FRAMES_EVAL=0.05989072623
LAMBDA_ACTIONS_EVAL=0.06929262371

EPS_TRAIN=0.03206952511
EPS_EVAL=0.04165084242
RADIUS_GW=0.002817559186

# --- Dataset parameters ---
DATASET="stone_pick_random/stone_pick_random_pixels_big"
FEATURE_NAME="pca_features"
STD_FEATS=false
SAVE_DIRECTORY="runs"
RUN="stone_pick_random_pixels_big"
VAL_FREQ=5

# --- General parameters ---
N_EPOCHS=14
BATCH_SIZE=2
N_FRAMES=147
LEARNING_RATE=0.002999715894
WEIGHT_DECAY=0.0000000221270297
LOG=true
VISUALIZE=true
SEED=0
RHO=0.09395031125
N_CLUSTERS=5
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
CMD="$CMD --visualize"
CMD="$CMD --log"

# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
