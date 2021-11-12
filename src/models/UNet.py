import torch
import torch.nn as nn


class double_conv(nn.Module):
    """
    Double Convolution layer with both 2 BN and Activation Layer in between
    Conv2d==>BN==>Activation==>Conv2d==>BN==>Activation
    """

    def __init__(self, in_channel, out_channel, drop_prob=0.0):
        super(double_conv, self).__init__()
        # initializing the model either with or without dropout

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p=drop_prob, inplace=False),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):
    """
    A maxpool layer followed by a Double Convolution.
    MaxPool2d==>double_conv.
    """

    def __init__(self, in_channel, out_channel, drop_prob):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), double_conv(in_channel, out_channel, drop_prob=drop_prob)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up_sample(nn.Module):
    def __init__(self, in_channel, out_channel, drop_prob):
        super(up_sample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.double_conv = double_conv(in_channel, out_channel, drop_prob=drop_prob)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    """Main Unet Model"""

    def __init__(self, model_configs):
        super(UNet, self).__init__()
        in_channel = model_configs["input_channels"]
        out_channel = model_configs["output_size"]
        drop_prob = model_configs["dropout_probabilities"][0]

        ## DownSampling Block
        self.down_block1 = double_conv(in_channel, 16)
        self.down_block2 = down_conv(16, 32, drop_prob)
        self.down_block3 = down_conv(32, 64, drop_prob)
        self.down_block4 = down_conv(64, 128, drop_prob)
        self.down_block5 = down_conv(128, 256, drop_prob)
        self.down_block6 = down_conv(256, 512, drop_prob)
        self.down_block7 = down_conv(512, 1024, drop_prob)
        ## UpSampling Block
        self.up_block1 = up_sample(1024 + 512, 512, drop_prob)
        self.up_block2 = up_sample(512 + 256, 256, drop_prob)
        self.up_block3 = up_sample(256 + 128, 128, drop_prob)
        self.up_block4 = up_sample(128 + 64, 64, drop_prob)
        self.up_block5 = up_sample(64 + 32, 32, drop_prob)
        self.up_block6 = up_sample(32 + 16, 16, drop_prob)
        self.up_block7 = nn.Conv2d(16, out_channel, 1)

    def forward(
        self,
        x,
        return_features: bool = False,
        **kwargs,
    ):
        """
        Forward of the contructed U-net
        Args:
            x: input image
            return_feaures (bool, default False): returns also the low level features
        Returns:
            image object passed through the networks with pixels classified as primary or background
        """
        # Down sampling
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        hidden_features = self.down_block7(x6)
        # Up sampling
        x8 = self.up_block1(hidden_features, x6)
        x9 = self.up_block2(x6, x5)
        x10 = self.up_block3(x9, x4)
        x11 = self.up_block4(x10, x3)
        x12 = self.up_block5(x11, x2)
        x13 = self.up_block6(x12, x1)
        out = self.up_block7(x13)

        if return_features:
            output_dict = {}
            output_dict["prediction"] = out
            output_dict["features"] = hidden_features
            return output_dict
        else:
            return out
