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

    def __init__(self, model_configs):
        """Defines a simple deterministic MLP
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
            nn.Conv2d(input_channels, layers[0], kernel_sizes[0], strides=1, padding=1),
            nn.ReLU(),
        )

        conv_layers = []
        for i in range(1, len(layers) - 1):
            layer = nn.Sequential(
                nn.Conv2d(
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
            nn.Dropout(p=dropout_probas[0]),
            nn.Flatten(),
            nn.Linear(width, 128),  # currently wrong, dimensions passed via config
            nn.ReLU(),
            nn.Dropout(p=dropout_probas[1]),
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
