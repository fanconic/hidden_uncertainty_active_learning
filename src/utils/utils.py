from torchvision.datasets import (
    MNIST,
    CIFAR10,
    CIFAR100,
    FashionMNIST,
    SVHN,
    Cityscapes,
    VOCSegmentation,
)
from ddu_dirty_mnist import DirtyMNIST
from src.data.segmentation_dataset import SegList

from src.models.MLP import MLP
from src.models.CNN import CNN
from src.models.BNN import BNN
from src.models.BCNN import BCNN
from src.models.MIR import MIR
from src.models.UNet import UNet
from src.models.resnet import ResNet
from src.models.DRNSeg import DRNSeg
from src.models.deeplab_v3plus import ModelDeepLabV3Plus

from src.active.heuristics import *
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt

CITYSCAPE_PALETTE = np.asarray(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)
CITYSCAPE_PALETTE = torch.Tensor(CITYSCAPE_PALETTE)


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
    elif "drn" in name:
        return DRNSeg(model_configs)
    elif "deeplab" in name:
        return ModelDeepLabV3Plus(model_configs)
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


def get_optimizer(model, config):
    """Resolve the optimizer according to the configs
    Args:
        model: model on which the optimizer is applied on
        config: configuration dict
    returns:
        optimizer
    """
    if config["optimizer"]["type"] == "SGD":
        optimizer = optim.SGD(
            model.optim_parameters(),
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            betas=config["optimizer"]["betas"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    return optimizer


def get_scheduler(optimizer, config):
    """Take the specified scheduler
    Args:
        optimizer: optimizer on which the scheduler is applied
        config: configuration dict
    returns:
        resovled scheduler
    """
    scheduler_name = config["training"]["scheduler"]
    assert scheduler_name in [
        "reduce_on_plateau",
        "step",
        "poly",
        "CosAnnWarmup",
    ], "scheduler not Implemented"

    if scheduler_name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["training"]["lr_reduce_factor"],
            patience=config["training"]["patience_lr_reduce"],
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"]["patience_lr_reduce"],
            gamma=config["training"]["lr_reduce_factor"],
        )
    elif scheduler_name == "poly":
        epochs = config["training"]["epochs"]
        poly_reduce = config["training"]["poly_reduce"]
        lmbda = lambda epoch: (1 - (epoch - 1) / epochs) ** poly_reduce
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)

    elif scheduler_name == "CosAnnWarmup":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["optimizer"]["T0"],
            T_mult=1,
            eta_min=config["optimizer"]["lr"] * 1e-2,
            last_epoch=-1,
        )

    else:
        scheduler = None
    return scheduler


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

    elif name == "dirty_mnist":
        train_ds = DirtyMNIST(
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

    elif name == "voc":
        train_ds = VOCSegmentation(
            "/srv/beegfs02/scratch/density_estimation/data/fanconic/",
            download=True,
            image_set="train",
            transform=train_transform,
            target_transform=train_target_transform,
        )
        test_ds = VOCSegmentation(
            "/srv/beegfs02/scratch/density_estimation/data/fanconic/",
            download=True,
            image_set="val",
            transform=test_transform,
            target_transform=test_target_transform,
        )

    elif name == "cityscapes_yu":
        train_ds = SegList(path, "train", transforms=train_transform)
        test_ds = SegList(path, "val", transforms=test_transform)

    else:
        raise NotImplemented

    return train_ds, test_ds


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha='center')
