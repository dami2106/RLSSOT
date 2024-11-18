#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SOTA

python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 --seed 0 --group main_results --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 -ls 512 128 40 -ua