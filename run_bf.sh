#!/bin/bash

actions=("cereals")
clusters=(4)
seed=0
gpu=0

for i in ${!actions[@]}; do
	python3 src/train.py -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 25 -f 8 --seed 0 -s --rho 0.5 -lat 0.05 -r 0.02 -ae 0.7 -at 0.4 -lr 1e-3 -wd 1e-4 -vf 5 --group main_results --wandb -v -ua 
done
