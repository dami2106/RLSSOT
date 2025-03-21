#!/bin/bash

python src/train.py -d desktop_assembly -at 0.3 -ae 0.6 -ua -lat 0.001 -r 0.02 -c 4 --rho 0.15 -f 20 --n-epochs 30 -ls 1345 256 40 -et 0.07 -ee 0.03 --seed 0 --group main_results --wandb -v -ac all
