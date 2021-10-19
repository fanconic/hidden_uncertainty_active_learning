from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from src.models.MLP import MLP
from src.models.CNN import CNN
from src.models.BNN import BNN
from src.models.BCNN import BCNN
from src.models.MIR import MIR

from src.active.heuristics import *


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
    elif name == "MIR":
        return MIR(model_configs)
    else:
        raise NotImplemented


def get_heuristic(heuristic_name):
    """Get the right heuristic
    args:
        heuristic_name (str): name of the heuristic
    returns:
        Heuristic
    """
    heuristic_name = heuristic_name.lower()
    if heuristic_name == "bald":
        return BALD()
    elif heuristic_name == "batchbald":
        return BatchBALD(4000)
    elif heuristic_name == "variance":
        return Variance()
    elif heuristic_name == "random":
        return Random(seed=42)
    elif heuristic_name == "entropy":
        return Entropy()
    elif heuristic_name == "margin":
        return Margin()
    elif heuristic_name == "certainty":
        return Certainty()
    elif heuristic_name == "precomputed":
        return Precomputed()
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
