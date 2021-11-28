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
- `active_learning_classification.py`: Main training loop for classification active learning
- `active_learning_segmentation.py`: Main training loop for segmentation active learning
- `config.yaml`: Config yaml file, which has all the experimental settings.


### Source Code Directory Tree
```
.
└── src                 # Source code            
    ├── layers              # Single Neural Network layers
    ├── models              # Neural Network Models
    ├── active              # Folder with functions for active learning
    ├── data                # Folder with data processing parts
    └── utils               # Useful functions, such as metrics, losses, etc
├── experiment_configs  # All the various configuration files for the experiments
└── experiment_outputs  # All outputs files of the experiments        
```


## How to train on CVL's GPU Cluster
```
bash gpu_experiments.sh <name_of_your_experiment>
```

## Prepare data
## Prepare Cityscapes training data

### Step 1

After you get a vanilla version of Cityscape data label maps, first convert the original segmentation label ids to one of 19 training ids:

```
python3 src/data/prepare_data.py <cityscape folder>/gtFine/
```
(Despite if the process is queued in SLURM, as soon as it starts it will extract the copied config.yaml file)

### Step 2

- Copy and run `create_lists.sh` in cityscape data folder, containing `gtFine` and `leftImg8bit` to create image and label lists.

## Contributors
- Claudio Fanconi - fanconic@ethz.ch
- Janis Postels - jpostels@vision.ee.ethz.ch

## References
- https://github.com/kumar-shridhar/PyTorch-BayesianCNN
- https://github.com/ElementAI/baal
- https://github.com/cameronccohen/deep-ensembles/
- https://github.com/fyu/drn/