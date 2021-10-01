from torch import nn
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from src.models import *


def get_model(model_configs):
    """create a torch model with the given configs
    args:
        model_configs: dict containing the model specific parameters
    returns:
        torch model
    """
    name = model_configs["name"]

    if name == "MLP":
        return MLP(model_configs)
    elif name == "CNN":
        return CNN(model_configs)
    elif name == "BNN":
        return BNN(model_configs)
    elif name == "BCNN":
        return BCNN(model_configs)
    else:
        raise NotImplemented


def load_data(name="mnist", train_transform=None, test_transform=None):
    """Load dataset
    Args:
        name (default "MNIST"): string name of the dataset
        train_transform (default None): training images transform
        test_transform (default None): test images transform
    Returns:
        train dataset, test dataset
    """

    if name == "cifar10":
        train_ds = CIFAR10("/tmp", train=True, transform=train_transform, download=True)
        test_ds = CIFAR10("/tmp", train=False, transform=test_transform, download=True)

    elif name == "cifar100":
        train_ds = CIFAR100(
            "/tmp", train=True, transform=train_transform, download=True
        )
        test_ds = CIFAR100("/tmp", train=False, transform=test_transform, download=True)

    elif name == "mnist":
        train_ds = MNIST("/tmp", train=True, transform=train_transform, download=True)
        test_ds = MNIST("/tmp", train=False, transform=test_transform, download=True)

    else:
        raise NotImplemented

    return train_ds, test_ds


# Taken from https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/layers/misc.py
class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, "set_flag"):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, "kl_loss"):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):
    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
