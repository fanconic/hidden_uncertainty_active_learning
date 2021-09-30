# Hidden Uncertainty for Active Learning
Implementation of Hidden Uncertainty for Active Learning in PyTorch.

## Introduction
TBC

## Setup
### Installation
Clone this repository.
```bash
$ git clone https://github.com/fanconic/hidden_uncertainty_active_learning
$ cd hidden_uncertainty_active_learning
```

I suggest to create a virtual environment and install the required packages.
```bash
$ conda create --name pytcu10 pytorch torchvision cudatoolkit=10.1 --channel pytorch
$ conda activate pytcu10
$ conda install --file requirements.txt
```

### Repository Structure
- `run.sh`: Runs the training script, include conda environment
- `train.py`: Main training loop for
- `config.yaml`: Config yaml file, which has all the experimental settings.


### Source Code Directory Tree
```
.
└── src                 # Source code
    ├──              
    ├──               
    └── 
├── experiment_configs  # All the various configuration files for the experiments
└── experiment_outputs  # All outputs files of the experiments        
```


## How to train on CVL's GPU Cluster
```
sbatch --output=log/%j.out --gres=gpu:1 --mem=30G ./run.sh
```

## Contributors
- Claudio Fanconi - fanconic@ethz.ch
- Janis Postels - jpostels@vision.ee.ethz.ch

## References
- https://github.com/kumar-shridhar/PyTorch-BayesianCNN
