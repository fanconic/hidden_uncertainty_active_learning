# BBBConv2d taken from https://github.com/kumar-shridhar/PyTorch-BayesianCNN

import torch
import torch.nn as nn
import numpy as np

from src.layers.BBBLinear import BBBLinear
from src.layers.BBBConv2d import BBBConv2d


class BayesianConvolutionLayer(torch.nn.Module):
    """
    Module implementing a single  bayesian convolutional layer, consisting of the convolution and non-linearity
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
            BBBConv2d(
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


class BCNN(torch.nn.Module):
    """
    Module implementing a Bayesian feedforward neural network using
    BayesianLayer objects.
    """

    def __init__(
        self, input_size, output_size, num_layers, width, use_bias=True, dropout=0.0
    ):
        """Defines a Bayesian Neural Network, with a distribution over the weights
        Args:
            input_size: size of the input data
            output_size: size of the output data
            num_layers: number of hidden layers
            width: number of neurons per hidden layer
            use_bias (default True): check if biases are used or not
            dropout (default): dropout probability
        """
        super().__init__()
        self.dropout = dropout
        self.use_bias = use_bias
        assert self.dropout < 1 and self.dropout >= 0
        conv_layers = [
            BayesianConvolutionLayer(input_size, width) for _ in range(num_layers)
        ]
        output_layer = BBBLinear(width, output_size)
        layers = [
            *conv_layers,
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25, inplace=False),
            nn.Flatten(),
            BBBLinear(width, width),  # currently wrong, dimensions passed via config
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            output_layer,
        ]
        self.net = nn.Sequential(*layers)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the neural network
        Args:
            x: input data
        Returns:
            out: logit data
        """
        return self.net(x)

    def predict_class_probs(self, x, num_forward_passes=50):
        """Forward pass through the neural network and predicts class probability, via multiple forward passes
        Args:
            x: input data
            num_forward_passes: number of forward passes
        Returns:
            out: class probabilities
        """
        ys = []
        for _ in range(num_forward_passes):
            y_hat = self.net(x)
            softmax_layer = nn.Softmax(dim=1)
            y_hat = softmax_layer(y_hat)

            ys.append(y_hat.detach().numpy())

        ys = np.array(ys)
        probs = ys.mean(axis=0)
        return torch.Tensor(probs)

    def kl_loss(self):
        """Computes the KL divergence loss for all layers.
        returns:
            sum of kl divergence loss over all layers
        """
        kl_sum = 0

        for layer in self.net.modules():
            if isinstance(layer, BBBLinear) or isinstance(layer, BBBConv2d):
                kl_sum += layer.kl_divergence()
        return kl_sum
