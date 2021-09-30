import torch
from torch import nn
from torch.nn import functional as F


class ConvolutionLayer(torch.nn.Module):
    """
    Module implementing a single convolutional layer, consisting of the convolution and non-linearity
    """

    def __init__(
        self, input_channels, output_channels, kernel_size=3, strides=1, padding=1
    ):
        """Defines a Bayesian Layer, with distribution over its weights
        Args:
            input_channels: size of the input data
            output_channels: size of the output data
            kernel_size (default 3): size of the convolutional filters
            strides (default 1): stride size of the convolution operation
            padding (default 1): padding size at the edges
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                strides=strides,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        """Forward pass through the BNN
        Args:
            inputs: input data
        returns:
            processed data
        """
        return self.conv(inputs)


class CNN(torch.nn.Module):
    """
    Simple module implementing a convolutional neural network with
    """

    def __init__(self, input_size, output_size, num_layers, width, drop_prob=0):
        """Defines a simple deterministic MLP
        Args:
            input_size: size of the input data
            output_size: size of the output data
            num_layers: number of hidden layers
            width: number of neurons per hidden layer
            drop_prob (default 0): dropout probability
        """
        super().__init__()
        conv_layers = [ConvolutionLayer(input_size, width) for _ in range(num_layers)]
        output_layer = nn.Linear(width, output_size)
        layers = [
            *conv_layers,
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25, inplace=False),
            nn.Flatten(),
            nn.Linear(width, width),  # currently wrong, dimensions passed via config
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            output_layer,
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the neural network
        Args:
            x: input data
        Returns:
            out: logit data
        """
        out = self.net(x)
        return out

    def predict_class_probs(self, x):
        """Forward pass through the neural network and predicts class probability
        Args:
            x: input data
        Returns:
            out: class probabilities
        """
        probs = F.softmax(self.forward(x), dim=1)
        return probs
