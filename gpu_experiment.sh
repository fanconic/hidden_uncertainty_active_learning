#!/bin/bash
echo "Running Experiment $1"
sbatch --output=experiment_outputs/week4/$1.out --gres=gpu:1 --mem=30G ./run.sh && cp config.yaml experiment_configs/week4/$1.yaml