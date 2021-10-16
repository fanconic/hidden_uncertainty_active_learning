import torch
from torch import nn
from torch.nn import functional as F


class CNN(torch.nn.Module):
    """
    Simple module implementing a convolutional neural network with
    """

    def __init__(self, model_configs):
        """Defines a simple deterministic MLP
        Args:
            model_configs: dict of configuration for the model
        """
        super().__init__()
        input_channels = model_configs["input_channels"]
        output_size = model_configs["output_size"]
        input_height = model_configs["input_height"]
        input_width = model_configs["input_width"]
        layers = model_configs["hidden_layers"]
        kernel_sizes = model_configs["kernel_sizes"]
        dropout_probas = model_configs["dropout_probabilities"]
        assert len(layers) == len(kernel_sizes)
        assert len(dropout_probas) == 2

        input_layer = torch.nn.Sequential(
            nn.Conv2d(
                input_channels, layers[0], kernel_sizes[0], stride=1, padding="same"
            ),
            nn.ReLU(),
        )

        conv_layers = []
        for i in range(1, len(layers) - 1):
            layer = nn.Sequential(
                nn.Conv2d(
                    layers[i],
                    layers[i + 1],
                    kernel_sizes[i + 1],
                    stride=1,
                    padding="same",
                ),
                nn.ReLU(),
            )
            conv_layers.append(layer)

        output_layer = nn.Linear(128, output_size)
        all_layers = [
            input_layer,
            *conv_layers,
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_probas[0]),
            nn.Flatten(),
            nn.Linear((input_height // 2) * (input_width // 2) * layers[-1], 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_probas[1]),
            output_layer,
        ]
        self.net = nn.Sequential(*all_layers)

    def forward(self, x, **kwargs):
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
