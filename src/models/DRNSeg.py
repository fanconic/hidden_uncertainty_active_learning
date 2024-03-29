import torch
from torch import nn
import math

import src.models.DRN as drn


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_configs, name=None):
        super(DRNSeg, self).__init__()
        # extract from model configs
        model_name = model_configs["name"] if name is None else name
        classes = model_configs["output_size"]
        pretrained_model = model_configs["pretrained_model"]
        pretrained = model_configs["pretrained"]
        use_torch_up = model_configs["use_torch_up"]
        self.dropout = model_configs["dropout_probabilities"][0]

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, dropout=self.dropout
        )
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes, kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(
                classes,
                classes,
                16,
                stride=8,
                padding=4,
                output_padding=0,
                groups=classes,
                bias=False,
            )
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x, **kwargs):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)

        return_features = kwargs.pop("return_features", False)
        if return_features:
            return y, x
        else:
            return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
