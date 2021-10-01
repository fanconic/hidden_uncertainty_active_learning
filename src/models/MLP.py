import torch
from torch import nn
from torch.nn import functional as F


class MLP(torch.nn.Module):
    """
    Simple module implementing a feedforward neural network with
    num_layers layers of size width and input of size input_size.
    """

    def __init__(self, model_configs):
        """Defines a simple deterministic MLP
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
        assert len(dropout_probas) == len(layers)

        input_layer = torch.nn.Sequential(
            nn.Flatten(), nn.Linear(input_size, layers[0]), nn.ReLU()
        )

        hidden_layers = []
        for i in range(len(layers) - 1):
            layer = nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                nn.ReLU(),
                nn.Dropout(p=dropout_probas[i]),
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
