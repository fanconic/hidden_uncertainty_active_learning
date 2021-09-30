from torch import nn
from keras.datasets import cifar10, cifar100, reuters, imdb, mnist


def load_data(name="mnist"):
    """Load dataset
    Args:
        name (default "MNIST"): string name of the dataset
    Returns:
        X_train, y_train, X_test, y_test
    """

    if name == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        return (X_train, y_train), (X_test, y_test)

    elif name == "cifar100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data("fine")
        return (X_train, y_train), (X_test, y_test)

    elif name == "reuters":
        (X_train, y_train), (X_test, y_test) = reuters.load_data()
        return (X_train, y_train), (X_test, y_test)

    elif name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return (X_train, y_train), (X_test, y_test)

    elif name == "imdb":
        (X_train, y_train), (X_test, y_test) = imdb.load_data()
        return (X_train, y_train), (X_test, y_test)

    else:
        raise NotImplementedError


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
