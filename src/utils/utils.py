from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from src.models.MLP import MLP
from src.models.CNN import CNN
from src.models.BNN import BNN
from src.models.BCNN import BCNN
from src.models.MIR import MIR


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
