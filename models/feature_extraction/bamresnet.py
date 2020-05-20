"""
    BAM-ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.
"""

__all__ = ['BamResNet', 'bam_resnet18', 'bam_resnet34', 'bam_resnet50', 'bam_resnet101', 'bam_resnet152']

import os
import torch.nn as nn
import torch.nn.init as init
from models.attention_map import BamBlock
from models.common import conv1x1_block, conv3x3_block, conv7x7_block


class DenseBlock(nn.Module):
    """
    Standard dense block with Batch normalization and ReLU activation.

    Parameters:
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    """

    def __init__(self,
                 in_features,
                 out_features):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features)
        self.bn = nn.BatchNorm1d(num_features=out_features)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class ResInitBlock(nn.Module):
    """
    ResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResInitBlock, self).__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResBottleneck(nn.Module):
    """
    ResNet bottleneck block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 conv1_stride=False,
                 bottleneck_factor=4):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1))
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            padding=padding,
            dilation=dilation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResUnit(nn.Module):
    """
    ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 bottleneck=True,
                 conv1_stride=False):
        super(ResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv1_stride=conv1_stride)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class BamResUnit(nn.Module):
    """
    BAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(BamResUnit, self).__init__()
        self.use_bam = (stride != 1)

        if self.use_bam:
            self.bam = BamBlock(channels=in_channels)
        self.res_unit = ResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck=bottleneck,
            conv1_stride=False)

    def forward(self, x):
        if self.use_bam:
            x = self.bam(x)
        x = self.res_unit(x)
        return x


class BamResNet(nn.Module):
    """
    BAM-ResNet model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(BamResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), BamResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        # self.features.add_module('final_pool', nn.AvgPool2d(
        #     kernel_size=7,
        #     stride=1))

        # self.features.add_module('final_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.output(x)
        return x


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create BAM-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported BAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BamResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def bam_resnet18(cfg, **kwargs):
    """
    BAM-ResNet-18 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="bam_resnet18", pretrained=cfg.pretrained, **kwargs)


def bam_resnet34(cfg, **kwargs):
    """
    BAM-ResNet-34 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="bam_resnet34", pretrained=cfg.pretrained, **kwargs)


def bam_resnet50(cfg, **kwargs):
    """
    BAM-ResNet-50 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="bam_resnet50", pretrained=cfg.pretrained, **kwargs)


def bam_resnet101(cfg, **kwargs):
    """
    BAM-ResNet-101 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="bam_resnet101", pretrained=cfg.pretrained, **kwargs)


def bam_resnet152(cfg, **kwargs):
    """
    BAM-ResNet-152 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="bam_resnet152", pretrained=cfg.pretrained, **kwargs)

# class ATBamResnet(nn.Module):
#     def __init__(self, num_classes, pretrained, attention_type):
#         super(ATBamResnet, self).__init__()
#         self.bam_resnet = bam_resnet50(pretrained=pretrained)
#         self.attention = AttentionMap(type=attention_type, num_channels=2048)
#         self.dropout = nn.Dropout(0.5)
#         self.last_linear = nn.Linear(2048, num_classes)
#
#     def logits(self, features):
#         x = F.adaptive_avg_pool2d(features, (1, 1))
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.last_linear(x)
#         return x
#
#     def forward(self, x):
#         x = self.bam_resnet.features(x)
#         x = self.attention(x)
#         x = self.logits(x)
#         return x
#
#
# def _test():
#     import torch
#
#     pretrained = True
#
#     models = [
#         bam_resnet18,
#         bam_resnet34,
#         bam_resnet50,
#         bam_resnet101,
#         bam_resnet152,
#     ]
#
#     for model in models:
#         net = model(pretrained=pretrained)
#
#         # net.train()
#         net.eval()
#         weight_count = _calc_width(net)
#         print("m={}, {}".format(model.__name__, weight_count))
#         assert (model != bam_resnet18 or weight_count == 11712503)
#         assert (model != bam_resnet34 or weight_count == 21820663)
#         assert (model != bam_resnet50 or weight_count == 25915099)
#         assert (model != bam_resnet101 or weight_count == 44907227)
#         assert (model != bam_resnet152 or weight_count == 60550875)
#
#         x = torch.randn(1, 3, 320, 320)
#         y = net(x)
#         y.sum().backward()
#         assert (tuple(y.size()) == (1, 1000))
#
#
# if __name__ == "__main__":
#     _test()
