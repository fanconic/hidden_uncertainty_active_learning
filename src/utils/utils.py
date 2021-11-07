from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST, SVHN
from src.models.MLP import MLP
from src.models.CNN import CNN
from src.models.BNN import BNN
from src.models.BCNN import BCNN
from src.models.MIR import MIR
from src.models.resnet import ResNet

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
    elif "resnet" in name:
        return ResNet(model_configs)
    else:
        raise NotImplemented


def get_heuristic(heuristic_name, random_state=0, shuffle_prop=0.0):
    """Get the right heuristic
    args:
        heuristic_name (str): name of the heuristic
        random_state (int): random state integer
        shuffle_prop (float): proportion that is shuffled
    returns:
        Heuristic
    """
    heuristic_name = heuristic_name.lower()
    if heuristic_name == "bald":
        return BALD(shuffle_prop=shuffle_prop)
    elif heuristic_name == "batchbald":
        return BatchBALD(4000, shuffle_prop=shuffle_prop)
    elif heuristic_name == "variance":
        return Variance()
    elif heuristic_name == "random":
        return Random(seed=random_state, shuffle_prop=shuffle_prop)
    elif heuristic_name == "entropy":
        return Entropy(shuffle_prop=shuffle_prop)
    elif heuristic_name == "margin":
        return Margin(shuffle_prop=shuffle_prop)
    elif heuristic_name == "certainty":
        return Certainty(shuffle_prop=shuffle_prop)
    elif heuristic_name == "precomputed":
        return Precomputed(shuffle_prop=shuffle_prop)
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

    name = name.lower()

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

    elif name == "fashion_mnist":
        train_ds = FashionMNIST(
            "/tmp", train=True, transform=train_transform, download=True
        )
        test_ds = FashionMNIST(
            "/tmp", train=False, transform=test_transform, download=True
        )

    elif name == "svhn":
        train_ds = SVHN("/tmp", split="train", transform=train_transform, download=True)
        test_ds = SVHN("/tmp", split="test", transform=test_transform, download=True)

    else:
        raise NotImplemented

    return train_ds, test_ds
