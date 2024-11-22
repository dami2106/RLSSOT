#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SOTA

python3 train.py -d desktop_assembly -ac all -c 3 -ne 10 --seed 0 --group testing --rho 0.15 -lat 0.11 -vf 5 -lr 1e-3 -wd 1e-4  -ls 11 128 40 -ua -f 4 -v -w -km
# python3 train.py -d desktop_assembly -ac all -c 3 -ne 30 --seed 0 --group main_results_2 --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 -ls 11 128 40 -ua -f 4 -v -w