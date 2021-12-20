#!/bin/bash
source /itet-stor/fanconic/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu10

python3 -u active_learning_classification.py --config_path experiment_configs/week10/$1.yaml