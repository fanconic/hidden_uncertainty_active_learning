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
        input_layer = torch.nn.Sequential(
            BBBLinear(input_size, width, dropout=self.dropout, bias=self.use_bias),
            nn.ReLU(),
        )
        hidden_layers = [
            nn.Sequential(
                BBBLinear(width, width, dropout=self.dropout, bias=self.use_bias),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        ]
        output_layer = BBBLinear(width, output_size, bias=self.use_bias)
        layers = [input_layer, *hidden_layers, output_layer]
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
            if isinstance(layer, BBBLinear):
                kl_sum += layer.kl_divergence()
        return kl_sum
