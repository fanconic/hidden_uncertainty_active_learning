from torchvision.datasets import (
    MNIST,
    CIFAR10,
    CIFAR100,
    FashionMNIST,
    SVHN,
    Cityscapes,
)
from src.models.MLP import MLP
from src.models.CNN import CNN
from src.models.BNN import BNN
from src.models.BCNN import BCNN
from src.models.MIR import MIR
from src.models.UNet import UNet
from src.models.resnet import ResNet

from src.active.heuristics import *


def get_model(model_configs):
    """create a torch model with the given configs
    args:
        model_configs: dict containing the model specific parameters
    returns:
        torch model
    """
    name = model_configs["name"].lower()

    if name == "mlp":
        return MLP(model_configs)
    elif name == "cnn":
        return CNN(model_configs)
    elif name == "bnn":
        return BNN(model_configs)
    elif name == "bcnn":
        return BCNN(model_configs)
    elif name == "mir":
        return MIR(model_configs)
    elif "resnet" in name:
        return ResNet(model_configs)
    elif "unet" in name:
        return UNet(model_configs)
    else:
        raise NotImplemented


def get_heuristic(heuristic_name, random_state=0, shuffle_prop=0.0, reduction="none"):
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
        return BALD(shuffle_prop=shuffle_prop, reduction=reduction)
    elif heuristic_name == "batchbald":
        return BatchBALD(4000, shuffle_prop=shuffle_prop, reduction=reduction)
    elif heuristic_name == "variance":
        return Variance()
    elif heuristic_name == "random":
        return Random(seed=random_state, shuffle_prop=shuffle_prop, reduction=reduction)
    elif heuristic_name == "entropy":
        return Entropy(shuffle_prop=shuffle_prop, reduction=reduction)
    elif heuristic_name == "margin":
        return Margin(shuffle_prop=shuffle_prop, reduction=reduction)
    elif heuristic_name == "certainty":
        return Certainty(shuffle_prop=shuffle_prop, reduction=reduction)
    elif heuristic_name == "precomputed":
        return Precomputed(shuffle_prop=shuffle_prop, reduction=reduction)
    else:
        raise NotImplemented


def load_data(
    name="mnist",
    train_transform=None,
    test_transform=None,
    path="/tmp",
    train_target_transform=None,
    test_target_transform=None,
):
    """Load dataset
    Args:
        name (default "MNIST"): string name of the dataset
        train_transform (default None): training images transform
        test_transform (default None): test images transform
        path (str, default "/tmp"): sting of the path to the dataset
        train_target_transform (default None): training labels transforms
        test_target_transform (default None): test labels transforms
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

    elif name == "cityscapes":
        train_ds = Cityscapes(
            path,
            split="train",
            target_type="semantic",
            transform=train_transform,
            target_transform=train_target_transform,
        )
        # in the City Scapes dataset, the test data set corresponds to the original validation dataset.
        # The new validation dataset is computed by taking a split
        test_ds = Cityscapes(
            path,
            split="val",
            target_type="semantic",
            transform=test_transform,
            target_transform=test_target_transform,
        )

        return train_ds, test_ds

    else:
        raise NotImplemented

    return train_ds, test_ds
