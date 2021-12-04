#!/bin/bash
echo "Running Experiment $1"
cp config.yaml experiment_configs/week9/$1.yaml
sbatch --output=experiment_outputs/week9/$1.out --gres=gpu:1 --mem=30G ./run.sh $1