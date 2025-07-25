#!/bin/bash

# --- Configuration ---
# Define the main task name (e.g., wsws_static)
TASK_NAME="wsws_static"

# Define the number of clusters
CLUSTERS=2


# Exit immediately if a command exits with a non-zero status.
set -e

# Define the subfolder suffixes to process
# Note: These correspond to the *end* part of the subfolder names like task_name_pixels, task_name_symbolic, etc.
# SUBFOLDER_SUFFIXES=("pixels" "pixels_big" "symbolic" "symbolic_big")
SUBFOLDER_SUFFIXES=("symbolic_big")

echo "Starting tuning runs for task: $TASK_NAME with $CLUSTERS clusters"
echo "=============================================================="

# Loop through each subfolder suffix    
for SUFFIX in "${SUBFOLDER_SUFFIXES[@]}"; do

    # Construct the full subfolder name (e.g., wsws_static_symbolic)
    SUBFOLDER_NAME="${TASK_NAME}_${SUFFIX}"

    # Construct the full path to the specific dataset/directory
    # Dataset path format: task_name/task_name_suffix
    DATASET_PATH="${TASK_NAME}/${SUBFOLDER_NAME}"


    # Determine feature name based on suffix
    if [[ "$SUFFIX" == *"symbolic"* ]]; then
        FEATURE_NAME="symbolic_obs_with_actions"
    elif [[ "$SUFFIX" == *"pixels"* ]]; then
        FEATURE_NAME="pca_features_with_actions"
    else
        echo "Warning: Unknown feature type for suffix '$SUFFIX'. Skipping."
        continue # Skip to the next iteration
    fi

    # Determine layer sizes based on the exact suffix
    case "$SUFFIX" in
        symbolic)
            LAYERS="1104 512 50"
            ;;
        symbolic_big)
            # NOTE: Using the same layers as 'symbolic' for 'symbolic_big'
            # as specific layers were only provided for symbolic, pixels, and pixels_big.
            # Adjust if 'symbolic_big' requires different layers.
            LAYERS="1104 512 50"
            ;;
        pixels)
            LAYERS="317 150 40"
            ;;
        pixels_big)
            LAYERS="667 300 40"
            ;;
        *)
            echo "Error: Unknown suffix '$SUFFIX' when determining layers. Skipping."
            continue # Skip to the next iteration
            ;;
    esac


    echo "--- Running configuration: $SUFFIX ---"
    echo "Dataset:       $DATASET_PATH"
    echo "Feature Name:  $FEATURE_NAME"
    echo "Clusters:      $CLUSTERS"
    echo "Layers:        \"$LAYERS\""
    echo "Command:"
    # Print the command that will be executed
    echo "python optuna_study_actions.py --dataset \"$DATASET_PATH\" --feature-name \"$FEATURE_NAME\" --clusters \"$CLUSTERS\" --layers \"$LAYERS\""
    echo "---------------------------------------"

    # Execute the python script
    python optuna_study_actions.py \
        --dataset "$DATASET_PATH" \
        --feature-name "$FEATURE_NAME" \
        --clusters "$CLUSTERS" \
        --layers "$LAYERS"

    echo "--- Finished configuration: $SUFFIX ---"
    echo # Add a newline for better readability between runs

done

echo "=============================================================="
echo "All tuning runs completed for task: $TASK_NAME"
