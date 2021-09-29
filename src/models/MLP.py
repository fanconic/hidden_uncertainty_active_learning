import torch
from torch import nn
from torch.nn import functional as F


class Densenet(torch.nn.Module):
    """
    Simple module implementing a feedforward neural network with
    num_layers layers of size width and input of size input_size.
    """

    def __init__(self, input_size, ouput_size, num_layers, width):
        """Defines a simple deterministic MLP
        Args:
            input_size: size of the input data
            output_size: size of the output data
            num_layers: number of hidden layers
            width: number of neurons per hidden layer
        """
        super().__init__()
        input_layer = torch.nn.Sequential(nn.Linear(input_size, width), nn.ReLU())
        hidden_layers = [
            nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(num_layers)
        ]
        output_layer = torch.nn.Linear(width, ouput_size)
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)

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
