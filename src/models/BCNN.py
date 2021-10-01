# BBBConv2d taken from https://github.com/kumar-shridhar/PyTorch-BayesianCNN

import torch
import torch.nn as nn
import numpy as np

from src.layers.BBBLinear import BBBLinear
from src.layers.BBBConv2d import BBBConv2d


class BCNN(torch.nn.Module):
    """
    Module implementing a Bayesian feedforward neural network using
    BayesianLayer objects.
    """

    def __init__(self, model_configs):
        """Defines a Bayesian Neural Network, with a distribution over the weights
        Args:
            model_configs: dict of configuration for the model
        """
        super().__init__()

        input_channels = model_configs["input_channels"]
        output_size = model_configs["output_size"]
        layers = model_configs["hidden_layers"]
        kernel_sizes = model_configs["kernel_sizes"]
        dropout_probas = model_configs["dropout_probabilities"]
        assert len(layers) == len(kernel_sizes)
        assert len(dropout_probas) == 2

        input_layer = torch.nn.Sequential(
            BBBConv2d(input_channels, layers[0], kernel_sizes[0], strides=1, padding=1),
            nn.ReLU(),
        )

        conv_layers = []
        for i in range(1, len(layers) - 1):
            layer = nn.Sequential(
                BBBConv2d(
                    layers[i],
                    layers[i + 1],
                    kernel_sizes[i + 1],
                    strides=1,
                    padding=1,
                ),
                nn.ReLU(),
            )
            conv_layers.append(layer)

        output_layer = nn.Linear(128, output_size)
        all_layers = [
            input_layer,
            *conv_layers,
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_probas[0]),
            nn.Flatten(),
            nn.Linear(width, 128),  # currently wrong, dimensions passed via config
            nn.ReLU(),
            nn.Dropout1d(p=dropout_probas[1]),
            output_layer,
        ]
        self.net = nn.Sequential(*all_layers)

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
