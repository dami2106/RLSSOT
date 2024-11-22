#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SOTA

python3 train.py -ac all -d desktop_assembly --n-epochs 30 --group testing_new_vis --ub-actions --visualize --wandb
