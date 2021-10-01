# Parts taken from https://github.com/ElementAI/baal/blob/master/experiments/mlp_mcdropout.py

import argparse
import yaml
import torch
from torchvision import transforms
from utils.utils import load_data, get_model
from data.dataset import ActiveLearningDataset
from src.layers.consistent_dropout import patch_module
from torch import nn, optim
import IPython


def main():
    """Main funciton to run"""

    use_cuda = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))

    # Load dataset
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    train_ds, test_ds = load_data(
        config["data"]["dataset"], train_transform, test_transform
    )

    al_dataset = ActiveLearningDataset(
        train_ds,
        pool_specifics={"transform": test_transform},
        random_state=config["random_state"],
    )
    al_dataset.label_randomly(200)  # Start with 200 items labelled.

    # Creates an MLP to classify MNIST
    model = get_model(config["model"])
    model = patch_module(model)  # Set dropout layers for MC-Dropout.
    if use_cuda:
        model = model.cuda()
    # wrapper = ModelWrapper(model=model, criterion=nn.CrossEntropyLoss())
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # We will use BALD as our heuristic as it is a great tradeoff between performance and efficiency.
    bald = BALD()


if __name__ == "__main__":
    main()
