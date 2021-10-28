# Taken from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
from numpy.lib.arraysetops import isin
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from src.models.resnet_base import load, DecodeBlock, resnet_layer
import copy
import warnings


class ResNet(torch.nn.Module):
    """
    Implementing a ResNet
    """

    def __init__(self, model_configs, name=None):
        """Defines a simple deterministic MLP
        Args:
            model_configs (dict): dict of configuration for the model
            name (str): name of the model backbone can be passed here, to overright the configs
        """
        super().__init__()
        self.input_channels = model_configs["input_channels"]
        self.output_size = model_configs["output_size"]
        self.input_height = model_configs["input_height"]
        self.input_width = model_configs["input_width"]
        self.pretrained = model_configs["pretrained"]
        self.model_name = model_configs["name"] if name is None else name
        self.dropout = model_configs["dropout_probabilities"][0]

        self.model = load(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.output_size,
            dropout=self.dropout,
        )

        self.feature_decoder = torch.nn.Sequential(*(list(self.model.children())[:-2]))
        self.avg_pool = self.model.avgpool
        self.output_layer = self.model.fc

    def forward(self, inputs, return_features=False, **kwargs):
        """Forward pass through the ResNet
        Args:
            inputs (torch.Tensor): input features
            return_features (bool): also returns the features before the fully connected layer
        returns:
            logis
        """
        out_features = self.feature_decoder(inputs)
        x = self.avg_pool(out_features)
        x = torch.flatten(x, 1)
        logits = self.output_layer(x)
        if not return_features:
            return logits
        else:
            return logits, out_features

    def predict_class_probs(self, x):
        """Forward pass through the neural network and predicts class probability
        Args:
            x: input data
        Returns:
            out: class probabilities
        """
        probs = F.softmax(self.forward(x), dim=1)
        return probs


class ResNetDecoder(torch.nn.Module):
    """
    Implementing the MLP decoder part for the hidden uncertainty model

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with stride=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    """

    def __init__(self, model_configs):
        """Defines a simple deterministic MLP
        Args:
            model_configs: dict of configuration for the model
        """
        super().__init__()

        self.input_height = model_configs["input_height"]
        self.input_width = model_configs["input_width"]
        self.input_channels = model_configs["input_channels"]

        self.feature_channels = model_configs["mir_configs"]["feature_dims"]
        self.dropout = model_configs["dropout_probabilities"][0]
        self.batchnorm = model_configs["mir_configs"]["decoder_bn"]
        self.num_res_blocks = model_configs["mir_configs"]["num_res_blocks"]

        if self.feature_channels == 64:
            num_stacks = 3
        elif self.feature_channels == 32:
            num_stacks = 2
        elif self.feature_channels == 16:
            num_stacks = 1
        elif self.feature_channels == 512:
            num_stacks = 4
        else:
            assert False, "Invalid Input Channel Shape! {}".format(
                str(self.feature_channels)
            )

        in_channels = self.feature_channels
        out_channels = self.feature_channels

        self.first_layer = resnet_layer(
            in_channels=in_channels,
            num_filters=out_channels,
            batch_normalization=self.batchnorm,
        )

        self.layers = []
        # Instantiate the stack of residual units
        for stack in range(num_stacks):
            decode_layer = DecodeBlock(
                in_channels,
                num_filters=out_channels,
                kernel_size=3,
                stride=1,
                batch_normalization=self.batchnorm,
                dropout=self.dropout,
            )

            if stack != 0:
                in_channels //= 2
            if stack != num_stacks - 1:
                out_channels //= 2

            self.layers.append(decode_layer)

        # upsample from 31x32 size to 32x32
        self.upsample = nn.Upsample(
            (self.input_height, self.input_width), mode="bilinear"
        )

        self.output_layer = resnet_layer(
            in_channels=out_channels,
            num_filters=self.input_channels,
            activation=False,
            batch_normalization=False,
        )

        self.net = nn.Sequential(
            self.first_layer, *self.layers, self.upsample, self.output_layer
        )

    def forward(self, x, **kwargs):
        """Forward pass through the neural network decoder
        Args:
            x: input data
        Returns:
            out: reconstructed input
        """
        out = self.net(x)
        return out
