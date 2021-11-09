#!/bin/bash
echo "Running Experiment $1"
cp config.yaml experiment_configs/week6/$1.yaml
sbatch --output=experiment_outputs/week6/$1.out --gres=gpu:1 --mem=30G ./run.sh --config_path experiment_configs/week6/$1.yaml