#!/usr/bin/env bash
set -euo pipefail

# Dataset path (edit if needed)
DATASET="../Craftax/Traces/stone_pickaxe_easy"

# Map each feature to its layer string
# pca_features_512   -> '512 256 64'
# pca_features_750   -> '750 312 64'
# pca_features_1000  -> '1000 432 64'
# resnet_features    -> '2048 800 64'
# clip_features      -> '512 256 64'
declare -A LAYERS_BY_FEATURE=(
  [pca_features_512]="512 256 64"
  [pca_features_750]="750 312 64"
  [pca_features_1000]="1000 432 64"
  [pca_features_2000]="2000 1024 256"
  [resnet_features]="2048 800 64"
  [clip_features]="512 256 64"
)

# Choose which features to submit (comment out any you don't want)
FEATURES=(
  pca_features_512
  pca_features_750
  pca_features_1000
  pca_features_2000
  resnet_features
  clip_features
)

# Submit one job per feature, passing variables via --export
for FEAT in "${FEATURES[@]}"; do
  LAYERS="${LAYERS_BY_FEATURE[$FEAT]:-}"
  if [[ -z "$LAYERS" ]]; then
    echo "No layer mapping for feature '$FEAT' — skipping." >&2
    continue
  fi

  # Filename can be derived from the feature (customize if needed)
  FILENAME="${FEAT//_/-}"

  # Submit and print the job ID
  JOBID=$(sbatch --parsable \
  --export=ALL,FEATURE="$FEAT",LAYERS="$LAYERS",FILENAME="$FILENAME",DATASET="$DATASET" \
  Scripts/tune_craftax_template.sbatch)

  echo "Submitted $FEAT  (layers: '$LAYERS')  as JobID $JOBID"
done