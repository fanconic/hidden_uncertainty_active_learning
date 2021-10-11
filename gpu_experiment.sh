#!/bin/bash
echo "Running Experiment $1"
sbatch --output=experiment_outputs/$1.out --gres=gpu:1 --mem=30G ./run.sh && cp config.yaml experiment_configs/$1.yaml