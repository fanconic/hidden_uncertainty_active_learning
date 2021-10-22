# Taken from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from src.models.resnet_base import load
import copy
import warnings


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
        self.dropout = model_configs["dropout_probabilities"][0]

        self.model = load(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.output_size,
            dropout=self.dropout,
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
