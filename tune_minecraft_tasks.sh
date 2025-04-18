#!/bin/bash

# --- Configuration ---
# Define the main task name (e.g., wsws_static)
TASK_NAME="wooden_pickaxe"

# Define the number of clusters
CLUSTERS=5

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting tuning run for task: $TASK_NAME with $CLUSTERS clusters"
echo "=============================================================="

# Set static subfolder info
DATASET_PATH="${TASK_NAME}"
FEATURE_NAME="pca_features"
LAYERS="4000 1500 200"

echo "--- Running configuration: $TASK_NAME ---"
echo "Dataset:       $DATASET_PATH"
echo "Feature Name:  $FEATURE_NAME"
echo "Clusters:      $CLUSTERS"
echo "Layers:        \"$LAYERS\""
echo "Command:"
echo "python optuna_study.py --dataset \"$DATASET_PATH\" --feature-name \"$FEATURE_NAME\" --clusters \"$CLUSTERS\" --layers \"$LAYERS\""
echo "---------------------------------------"

# Execute the python script
python optuna_study.py \
    --dataset "$DATASET_PATH" \
    --feature-name "$FEATURE_NAME" \
    --clusters "$CLUSTERS" \
    --layers "$LAYERS"

echo "--- Finished configuration: $SUFFIX ---"
echo "=============================================================="
echo "Tuning run completed for task: $TASK_NAME"
