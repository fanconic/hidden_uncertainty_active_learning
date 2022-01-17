# Hidden Uncertainty for Active Learning
Implementation of Hidden Uncertainty for Active Learning in PyTorch.

## Introduction
In this work, we compare the effectiveness of a newly proposed uncertainty quantification method called Hidden Uncertainty (HU) with existing deep uncertainty estimation methods for active learning (AL) in the context of image classification and semantic segmentation. We uncover critical shortcomings of HU when used for AL in both cases. For image classification, HU quantifies uncertainty poorly in the initial iterations of AL because its density estimator, used to quantify uncertainty, overfits all classes of the small number of training images. In semantic segmentation, HU’s density estimator overfits the minority classes of the training set, in the first iterations. Both shortcomings lead to poor uncertainty estimation and thus uninformative queries on the unlabelled dataset.\n
Subsequently, we propose three measures to address the above shortcomings in AL. To address the overfitting of HU in classification, we propose to (1) greedily reduce the dimensions of the hidden representation. Alternatively, (2) we replace the original HU density estimation method with a kernel density estimator (KDE) approximation. For overfitting to minority classes in segmentation, we additionally propose (3) to greedily reduce the dimensions of the hidden features depending on their predicted classes.\n
After incorporating our remedies, we find that (1) reduces overfitting in image classification, but not enough to result in significant improvements over HU when trained on less than 1’000 images. Notably, (2) outperforms HU and other established Deep Bayesian uncertainty estimation methods on MNIST, and scores competitively with them on the Cityscapes dataset after querying 1’000 or more images. Finally, we do not observe any performance improvement of (3) over HU.

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
- `run.sh`: Runs the training script, include conda environment, makes directly a copy
- `gpu_experiment.sh`: Runs the training script, include conda environment directly on the SLURM cluster
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

## Run experiments:
Run classification in terminal
```bash
python3 -u active_learning_classification.py --config_path ./config.yaml
```

Run segmentation in terminal
```bash
python3 -u active_learning_segmentation.py --config_path ./config.yaml
```

How to train on CVL's GPU Cluster (might have to change the command in ```run.sh```)
```bash
bash gpu_experiments.sh <name_of_your_experiment>
```

## Reproduce thesis experiments
To reproduce our experiments, you can run the commands that we have saved in /experiment_configs/experiment_commands.txt

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