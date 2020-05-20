##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNet variants"""
import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.attention_map import SEModule, SpatialCGNL, SAModule
from models.feature_extraction.splat import SplAtConv2d
from models.utils import gen_adj_num, gen_adj
from models.common import conv1x1

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'
_model_sha256 = {name: checksum for checksum, name in
                 [('528c19ca', 'resnest50'), ('22405ba7', 'resnest101'), ('75117900', 'resnest200'),
                  ('0cc87c48', 'resnest269'), ]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]


resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for name in
                      _model_sha256.keys()}

__all__ = ['ResNet', 'Bottleneck']


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1,
                 bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0,
                 last_gamma=False, use_se=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=3, stride=stride,
                                     padding=dilation, dilation=dilation, groups=cardinality,
                                     bias=False, radix=radix, rectify=rectified_conv,
                                     rectify_avg=rectify_avg, norm_layer=norm_layer,
                                     dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(group_width, group_width, kernel_size=3, stride=stride,
                                  padding=dilation, dilation=dilation, groups=cardinality,
                                  bias=False, average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride,
                                   padding=dilation, dilation=dilation, groups=cardinality,
                                   bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.use_se = use_se

        if use_se:
            self.se = SEModule(planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.use_se:
            out = self.se(out) + residual
        else:
            out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64, num_classes=1000,
                 dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False, avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0, last_gamma=False, use_se=False, in_channels=300,
                 word_file='/workspace/Projects/cxr/models/feature_extraction/diseases_embeddings.npy',
                 # word_file='diseases_embeddings.npy',
                 # word_file='/home/hoangvu/Projects/cxr/models/feature_extraction/diseases_embeddings.npy',
                 extract_fields='0,1,2,3,4,5', agree_rate=0.5, csv_path='',
                 norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.use_se = use_se

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False,
                           **conv_kwargs), norm_layer(stem_width), nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False,
                           **conv_kwargs), norm_layer(stem_width), nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1,
                           bias=False, **conv_kwargs), )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                                    **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2,
                                           norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,
                                           norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1,
                                           norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2,
                                           norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        num_classes = len(extract_fields.split(','))
        _adj = gen_adj_num(labels=extract_fields, agree_rate=agree_rate, csv_path=csv_path)
        self.adj = Parameter(torch.from_numpy(_adj).float())

        if not os.path.exists(word_file):
            word = np.random.randn(num_classes, 300)
            print('graph input: random')
        else:
            with open(word_file, 'rb') as point:
                word = np.load(point)
                print('graph input: loaded from {}'.format(word_file))
        self.word = Parameter(torch.from_numpy(word).float())

        self.gc0 = GraphConvolution(in_channels, 128, bias=True)
        self.gc1 = GraphConvolution(128, 256, bias=True)
        self.gc2 = GraphConvolution(256, 512, bias=True)
        self.gc3 = GraphConvolution(512, 1024, bias=True)
        self.gc4 = GraphConvolution(1024, 2048, bias=True)
        self.gc_relu = nn.LeakyReLU(0.2)
        self.gc_tanh = nn.Tanh()
        self.merge_conv0 = nn.Conv2d(num_classes, 128, kernel_size=1, stride=1, bias=False)
        self.merge_conv1 = nn.Conv2d(num_classes, 256, kernel_size=1, stride=1, bias=False)
        self.merge_conv2 = nn.Conv2d(num_classes, 512, kernel_size=1, stride=1, bias=False)
        self.merge_conv3 = nn.Conv2d(num_classes, 1024, kernel_size=1, stride=1, bias=False)
        self.conv1x1 = conv1x1(in_channels=2048, out_channels=num_classes, bias=True)
        # self.spatial_attention = SAModule(2048)
        # self.spatial_attention = SpatialCGNL(2048, 1024)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(
                        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True,
                                     count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True,
                                                    count_include_pad=False))
                down_layers.append(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                              bias=False))
            else:
                down_layers.append(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix,
                      cardinality=self.cardinality, bottleneck_width=self.bottleneck_width,
                      avd=self.avd, avd_first=self.avd_first, dilation=1, is_first=is_first,
                      rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg,
                      norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                      last_gamma=self.last_gamma, use_se=self.use_se))
        elif dilation == 4:
            layers.append(
                block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix,
                      cardinality=self.cardinality, bottleneck_width=self.bottleneck_width,
                      avd=self.avd, avd_first=self.avd_first, dilation=2, is_first=is_first,
                      rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg,
                      norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                      last_gamma=self.last_gamma, use_se=self.use_se))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, radix=self.radix, cardinality=self.cardinality,
                      bottleneck_width=self.bottleneck_width, avd=self.avd,
                      avd_first=self.avd_first, dilation=dilation,
                      rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg,
                      norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                      last_gamma=self.last_gamma, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, feature):
        adj = gen_adj(self.adj).detach()
        word = self.word.detach()

        feature = self.conv1(feature)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        x_raw = self.gc0(word, adj)
        x = self.gc_tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv0)

        feature = self.layer1(feature)
        x = self.gc_relu(x_raw)
        x_raw = self.gc1(x, adj)
        x = self.gc_tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv1)

        feature = self.layer2(feature)
        x = self.gc_relu(x_raw)
        x_raw = self.gc2(x, adj)
        x = self.gc_tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv2)

        feature = self.layer3(feature)
        x = self.gc_relu(x_raw)
        x_raw = self.gc3(x, adj)
        x = self.gc_tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv3)

        feature = self.layer4(feature)
        # feature = self.spatial_attention(feature)
        feature_raw = self.global_pool(feature)
        if self.drop is not None:
            feature_raw = self.drop(feature_raw)

        feature = feature_raw.view(feature_raw.size(0), -1)

        x = self.gc_relu(x_raw)
        x = self.gc4(x, adj)
        x = self.gc_tanh(x)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        y = self.conv1x1(feature_raw)
        y = y.view(y.size(0), -1)
        x = x + y
        return x


def gcn_resnest200(cfg=None, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False,
                   use_se=cfg.use_se, extract_fields=cfg.extract_fields, agree_rate=cfg.agree_rate,
                   csv_path=cfg.csv_path, **kwargs)
    # model = ResNet(Bottleneck, [3, 24, 36, 3], radix=2, groups=1, bottleneck_width=64,
    #                deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False,
    #                use_se=False, extract_fields='0,1,2,3,4,5', agree_rate=0.5,
    #                csv_path='D:/Dataset/Vinmec/Noise/train_sss.csv', **kwargs)
    if cfg.pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(resnest_model_urls['resnest200'], progress=True),
            strict=False)
    return model


def gcn_resnest101(cfg=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False,
                   use_se=cfg.use_se, extract_fields=cfg.extract_fields, agree_rate=cfg.agree_rate,
                   csv_path=cfg.csv_path, **kwargs)
    if cfg.pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(resnest_model_urls['resnest101'], progress=True),
            strict=False)
    return model


def gcn_resnest50(cfg=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                   stem_width=32, avg_down=True, avd=True, avd_first=False, use_se=cfg.use_se,
                   extract_fields=cfg.extract_fields, agree_rate=cfg.agree_rate,
                   csv_path=cfg.csv_path, **kwargs)
    if cfg.pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(resnest_model_urls['resnest50'], progress=True),
            strict=False)
    return model


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        middle_features = max(32, (in_features + out_features) // 16)
        self.weight1 = Parameter(torch.Tensor(in_features, middle_features))
        self.weight2 = Parameter(torch.Tensor(middle_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight1)
        support = torch.matmul(support, self.weight2)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(
            self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True, bias=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.bias is not None:
            h_prime = h_prime + self.bias

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(
            self.out_features) + ')'


def merge_gcn_residual(feature, x, merge_conv):
    feature_raw = feature
    feature = feature_raw.transpose(1, 2)
    feature = feature.transpose(2, 3).contiguous()
    feature = feature.view(-1, feature.shape[-1])
    reshape_x = x.transpose(0, 1)
    feature = torch.matmul(feature, reshape_x)
    feature = feature.view(feature_raw.shape[0], feature_raw.shape[2], feature_raw.shape[3], -1)
    feature = feature.transpose(2, 3)
    feature = feature.transpose(1, 2)

    feature = merge_conv(feature)

    return feature_raw + feature


if __name__ == "__main__":
    import torchsummary
    x = torch.randn([2, 3, 224, 224])
    model = gcn_resnest200(num_classes=6, word_file='diseases_embeddings.npy')
    logits = model(x)
    # print(torchsummary.summary(model, input_size=(3, 512, 512), device='cpu'))
    print(logits)

    # x = torch.randn([2, 2048, 7, 7])
    # word = torch.randn([6, 300])
    # adj = torch.randn([6, 6])  #
    # # gcn = GraphConvolution(in_features=300, out_features=256, bias=True)
    # gcn = GraphAttentionLayer(in_features=300, out_features=256, bias=True)
    # output = gcn(word, adj)
    # print(output)

    # feature = torch.randn([2, 128, 56, 56])  # x = torch.randn([11, 128])  # merge_conv = nn.Conv2d(11, 128, kernel_size=1, stride=1, bias=False)  #  # output = merge_gcn_residual(feature, x, merge_conv)  # print(output.size())
