import torch
from torch import nn
from torch.nn import functional as F


class ResNet(torch.nn.Module):
    """
    Implementing a ResNet
    """

    def __init__(self, model_configs):
        """Defines a simple deterministic MLP
        Args:
            model_configs: dict of configuration for the model
        """
        super().__init__()
        self.input_channels = model_configs["input_channels"]
        self.output_size = model_configs["output_size"]
        self.input_height = model_configs["input_height"]
        self.input_width = model_configs["input_width"]
        self.pretrained = model_configs["pretrained"]
        self.model_name = model_configs["name"]

        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.output_size,
        )

    def forward(self, inputs, **kwargs):
        """Forward pass through the ResNet
        Args:
            inputs (torch.Tensor): input features
        returns:
            logis
        """
        logits = self.model(inputs)
        return logits

    def predict_class_probs(self, x):
        """Forward pass through the neural network and predicts class probability
        Args:
            x: input data
        Returns:
            out: class probabilities
        """
        probs = F.softmax(self.forward(x), dim=1)
        return probs
