import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.layers.BBBLinear import BBBLinear


class BNN(torch.nn.Module):
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
        input_height = model_configs["input_height"]
        input_width = model_configs["input_width"]
        input_channels = model_configs["input_channels"]
        input_size = input_height * input_width * input_channels
        output_size = model_configs["output_size"]
        layers = model_configs["hidden_layers"]
        dropout_probas = model_configs["dropout_probabilities"]
        self.use_bias = model_configs["use_bias"]

        assert len(dropout_probas) == len(layers)

        input_layer = torch.nn.Sequential(
            nn.Flatten(),
            BBBLinear(input_size, layers[0], bias=self.use_bias),
            nn.ReLU(),
        )

        hidden_layers = []
        for i in range(len(layers) - 1):
            layer = nn.Sequential(
                BBBLinear(layers[i], layers[i + 1], bias=self.use_bias),
                nn.ReLU(),
                nn.Dropout1d(p=dropout_probas[i]),
            )
            hidden_layers.append(layer)

        output_layer = torch.nn.Linear(layers[-1], output_size)
        all_layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*all_layers)

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
            if isinstance(layer, BBBLinear):
                kl_sum += layer.kl_divergence()
        return kl_sum
