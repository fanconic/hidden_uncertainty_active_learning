import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class ModelDeepLabV3Plus(torch.nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.backbone = model_configs["mir_configs"]["backbone"]
        ch_out = model_configs["output_size"]

        self.encoder = Encoder(
            self.backbone,
            dropout=model_configs["dropout_probabilities"][0],
            pretrained=model_configs["pretrained"],
            zero_init_residual=True,
            replace_stride_with_dilation=(False, True, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(
            self.backbone
        )
        # ch_out_encoder_bottleneck = 512 default, out_stride = 16 default
        # ch_out_encoder_4x = 64 default, out_stride = 4 default
        self.aspp = ASPP(ch_out_encoder_bottleneck, 256)

        # 48 output channels was taken from the paper on page 8.
        # 32 output channels is also a possibility
        skip_4x_out_ch = 48

        # ch_out = semseg_num_classes + 1, out_stride = 4 default
        self.decoder = DecoderDeeplabV3p(256, ch_out_encoder_4x, skip_4x_out_ch, ch_out)

    def forward(self, x, **kwargs):
        input_resolution = (x.shape[2], x.shape[3])
        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # Note that we have 2 layers with output stride 16 if we use dilated convolution in last layers
        # In the dictionary the 16:256 gets overwritten by 16:512!

        lowest_scale = max(features.keys())
        features_lowest = features[lowest_scale]
        features_tasks = self.aspp(features_lowest)

        # Here we take the features from the encoder with output stride 4 (features[4])
        # This is dependent on the output stride of features from the ASPP module!
        predictions_4x, _ = self.decoder(features_tasks, features[4])

        predictions_1x = F.interpolate(
            predictions_4x, size=input_resolution, mode="bilinear", align_corners=False
        )
        # Number of channels here is semseg_num_classes and resolution same as input image
        return predictions_1x


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        dropout=0.0,
    ):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.dropout(out)
        out = self.relu(out)
        return out


_basic_block_layers = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            dropout = encoder_kwargs.pop("dropout", 0.0)
            pretrained = encoder_kwargs.pop("pretrained", False)
            progress = encoder_kwargs.pop("progress", True)
            model = resnet._resnet(
                name,
                BasicBlockWithDilation,
                _basic_block_layers[name],
                pretrained,
                progress,
                **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get(
            "replace_stride_with_dilation", (False, False, False)
        )
        if dropout < 1 and dropout > 0:
            model.layer1[0].dropout.p = dropout
            model.layer1[1].dropout.p = dropout
            model.layer2[0].dropout.p = dropout
            model.layer2[1].dropout.p = dropout
            model.layer3[0].dropout.p = dropout
            model.layer3[1].dropout.p = dropout
            model.layer4[0].dropout.p = dropout
            model.layer4[1].dropout.p = dropout

        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, skip_4x_out_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        self.conv1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, skip_4x_out_ch, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(skip_4x_out_ch),
            torch.nn.ReLU(),
        )

        # Add 256 + 48, as they will be concatena+ted
        self.features_to_predictions = torch.nn.Sequential(
            torch.nn.Conv2d(
                skip_4x_out_ch + bottleneck_ch, 256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, num_out_ch, kernel_size=3, stride=1, padding=1),
            # why not use relu here? -->  output used directly for predictions,
            # last channel corresponds to depth prediction
        )

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # DONE: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        # Low Level Features coming from the encoder
        low_level_features = self.conv1x1(features_skip_4x)

        # Features coming from the ASPP
        features_bottleneck_4x = F.interpolate(
            features_bottleneck,
            size=features_skip_4x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # concatenate low level features and upsampled --> channels = bottleneck_ch(256) + skip_4x_out_channels(48/32)
        features_4x = torch.cat([features_bottleneck_4x, low_level_features], dim=1)
        predictions_4x = self.features_to_predictions(features_4x)
        return predictions_4x, features_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation
    ):
        super().__init__(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=False,  # Possible variation: use bias here.
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        """
        In implementation form the paper to DeepLabV3Plus, BatchNorm and ReLu are
        only applied to the global average pooling but not on the atrous convolutions.
        See lines 426 - 540 in models.py on the referenced github in the paper.
        """


class ASPP(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, rates=(6, 12, 18)
    ):  # default rates from paper: (6,12,18)
        super().__init__()
        # DONE: Implement ASPP properly instead of the following
        self.branch1 = ASPPpart(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.branch2 = ASPPpart(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rates[0],
            dilation=rates[0],
        )
        self.branch3 = ASPPpart(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rates[1],
            dilation=rates[1],
        )
        self.branch4 = ASPPpart(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rates[2],
            dilation=rates[2],
        )
        self.branch5 = ASPPpart(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.conv_out = ASPPpart(
            out_channels * 5,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

    def forward(self, x):
        # Done: Implement ASPP properly instead of the following
        # Adapted from https://github.com/YudeWang/deeplabv3plus-pytorch/blob/master/lib/net/ASPP.py
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, keepdim=True)
        global_feature = torch.mean(global_feature, 3, keepdim=True)
        global_feature = self.branch5(global_feature)
        global_feature = F.interpolate(
            global_feature,
            size=(row, col),
            mode="bilinear",
            align_corners=True,  # True or False does not matter here since input has size 1.
        )

        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1
        )
        out = self.conv_out(feature_cat)
        return out


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed
