#!/bin/bash
echo "Running Experiment $1"
cp config.yaml experiment_configs/final_runs/$1.yaml
sbatch --output=experiment_outputs/final_runs/$1.out --gres=gpu:1 --mem=30G ./run.sh $1