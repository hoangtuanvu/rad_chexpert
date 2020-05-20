"""Summary
"""
import torch
from torch import nn

import torch.nn.functional as F
from .feature_extraction.vgg import (vgg19, vgg19_bn)
from .feature_extraction.bamresnet import bam_resnet50
from .feature_extraction.densenet import (densenet121, densenet161, densenet169, densenet201)
from .feature_extraction.inceptionresnetv2 import inceptionresnetv2
from .feature_extraction.inceptionv3 import inception_v3
from .feature_extraction.resnet import (resnet50, resnet101, resnet152)
from .feature_extraction.resnetfpn5 import (resnet34fpn5, resnet50fpn5, resnet101fpn5,
                                            resnet152fpn5)
from .feature_extraction.xception_dropout import xception_dropout
from .feature_extraction.dla import (dla34, dla60x, dla102x, dla102x2, dla169)
from .feature_extraction.bamdla import (bam_dla102x)
from .feature_extraction.se_resnet import (se_resnext50_32x4d, se_resnext101_32x4d, senet154)
from .feature_extraction.hubconf import (resnext50_32x4d_ssl, resnext50_32x4d_swsl,
                                         resnext101_32x16d_ssl, resnext101_32x16d_swsl,
                                         resnet18_ssl, resnet18_swsl)
from .feature_extraction.efficientnet import (tf_efficientnet_b0_ns, tf_efficientnet_b1_ns,
                                              tf_efficientnet_b2_ns, tf_efficientnet_b3_ns,
                                              tf_efficientnet_b4_ns)
from .feature_extraction.dpn import (dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131)
from .feature_extraction.resnest import (resnest50, resnest101, resnest200, resnest269)
from .feature_extraction.gcn_resnest import (gcn_resnest200, gcn_resnest101, gcn_resnest50)
from .common import conv1x1
from models.global_pool import GlobalPool
from models.attention_map import AttentionMap, ClassWisePool, WildcatPool2d
from models.utils import CBatchNorm2d, beta_mish, GraphConvolution, gen_adj_num, input_embedding, \
    gen_adj
from torch.nn import Parameter


class Classifier(nn.Module):
    """Summary

    Attributes:
        avai_backbones (TYPE): Description
        backbone (TYPE): Description
        backbone_type (TYPE): Description
        cfg (TYPE): Description
        expand (int): Description
        global_pool (TYPE): Description
    """

    def __init__(self, cfg, hparams=None):
        """Summary

        Args:
            cfg (TYPE): Description

        Raises:
            KeyError: Description
        """
        super(Classifier, self).__init__()
        self.cfg = cfg

        self.avai_backbones = self.get_backbones()
        if self.cfg.backbone not in self.avai_backbones.keys():
            raise KeyError(
                'Invalid backbone name. Received "{}", but expected to be one of {}'.format(
                    self.cfg.backbone, self.avai_backbones.keys()))

        self.backbone = self.avai_backbones[cfg.backbone][0](cfg)

        if self.cfg.bb_freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.backbone_type = self.avai_backbones[self.cfg.backbone][1]
        self.global_pool = GlobalPool(cfg)
        if hparams is not None and hparams.multi_cls:
            self.num_class = len(self.cfg.extract_fields.split(',')) + 1
        else:
            self.num_class = len(self.cfg.extract_fields.split(','))
        self.num_maps = 1

        if self.cfg.wildcat:
            self.num_maps = self.cfg.num_maps
            pooling = nn.Sequential()
            pooling.add_module('class_wise', ClassWisePool(self.num_maps))
            pooling.add_module('spatial', WildcatPool2d(1, 1, 0.7))
            self.spatial_pooling = pooling

        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3

        if cfg.attention_map.endswith('C'):
            self.expand *= 2

        if self.backbone_type in ['vgg', 'resnet_ssl']:
            out_features = 512
        elif self.backbone_type in ['tf_efficientnet_ns']:
            out_features = 1280
        elif self.backbone_type in ['tf_efficientnet_ns_b4']:
            out_features = 1792
        elif self.backbone_type in ['densenet']:
            out_features = self.backbone.num_features
        elif self.backbone_type in ['inception', 'resnet', 'xception', 'se_resnext', 'sk_resnext',
                                    'resnext_ssl', 'poly']:
            out_features = 2048
        elif self.backbone_type in ['light_cnn']:
            out_features = 128
        elif self.backbone_type in ['inceptionresnetv2']:
            out_features = 1536
        elif self.backbone_type in ['dla', 'dla_cbn']:
            out_features = self.backbone.channels[-1]
        elif self.backbone_type in ['shufflenet_v2']:
            out_features = 1024
        elif self.backbone_type in ['dpn832']:
            out_features = 832
        elif self.backbone_type in ['dpn2688']:
            out_features = 2688
        else:
            raise Exception('Unknown backbone type : {}'.format(self.cfg.backbone))

        if self.cfg.embedded_gcn and not self.cfg.within_kernel:
            # Class-wise pool
            self.num_maps = self.cfg.num_maps
            self.clw_pool = ClassWisePool(self.num_maps)

            # Graph CNN
            self.gc1 = GraphConvolution(self.cfg.embedded_size, self.cfg.num_gcn_maps)
            self.gc2 = GraphConvolution(self.cfg.num_gcn_maps, 1)
            _adj = gen_adj_num(labels=self.cfg.extract_fields, agree_rate=self.cfg.agree_rate)
            self.adj = Parameter(torch.from_numpy(_adj).float())
            self.relu = nn.LeakyReLU(0.2)

        self.out_features = out_features
        self._init_classifier()
        self._init_bn()
        self._init_attention_map()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_classifier(self):
        """Summary

        Raises:
            Exception: Description
        """
        if self.cfg.conv_fc:
            setattr(self, "fc",
                    conv1x1(self.out_features * self.expand, self.num_class * self.num_maps,
                            bias=True))
        else:
            setattr(self, "fc",
                    nn.Linear(self.out_features * self.expand, self.num_class * self.num_maps,
                              bias=True))

        classifier = getattr(self, "fc")
        if isinstance(classifier, nn.Conv2d):
            classifier.weight.data.normal_(0, 0.01)
            classifier.bias.data.zero_()

    def _init_bn(self):
        """Summary

        Raises:
            Exception: Description
        """
        setattr(self, "bn", nn.BatchNorm2d(self.out_features * self.expand))

    def _init_attention_map(self):
        """Summary

        Raises:
            Exception: Description
        """
        setattr(self, "attention_map", AttentionMap(self.cfg, self.out_features))

    def cuda(self, device=None):
        """Summary

        Args:
            device (None, optional): Description

        Returns:
            TYPE: Description
        """
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        # (N, C, H, W)
        feat_map = self.backbone(x)

        if self.cfg.mish:
            feat_map = beta_mish(feat_map, beta=self.cfg.beta_mish)

        if self.cfg.attention_map != "None":
            feat_map = self.attention_map(feat_map)

        if self.cfg.embedded_gcn:
            if self.cfg.within_kernel:
                logits = feat_map
            else:
                classifier = getattr(self, "fc")
                # apply class-wise pool
                feat_map = classifier(feat_map)
                feat = self.clw_pool(feat_map)
                bs = feat.size(0)
                c = feat.size(1)
                feat = feat.view((bs, c, -1))
                # print(feat.size())
                #
                adj = gen_adj(self.adj).detach()
                if self.cfg.transpose:
                    adj = torch.transpose(adj, 0, 1)

                embedded = self.gc1(feat, adj)
                embedded = self.relu(embedded)
                embedded = self.gc2(embedded, adj)

                logits = embedded.view(embedded.size(0), -1)
        else:
            classifier = getattr(self, "fc")
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map)

            if self.cfg.fc_bn:
                bn = getattr(self, "bn")
                feat = bn(feat)

            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            if self.cfg.conv_fc:
                if self.cfg.wildcat:
                    logits = classifier(feat)
                    logits = self.spatial_pooling(logits)
                else:
                    logits = classifier(feat).squeeze(-1).squeeze(-1)

            else:
                logits = classifier(feat.view(feat.size(0), -1))

        return logits

    @staticmethod
    def get_backbones():
        """Summary

        Returns:
            TYPE: Description
        """
        __factory = {'vgg19': [vgg19, 'vgg'], 'vgg19_bn': [vgg19_bn, 'vgg'],
                     'bam_resnet50': [bam_resnet50, 'resnet'],
                     'densenet121': [densenet121, 'densenet'],
                     'densenet161': [densenet161, 'densenet'],
                     'densenet169': [densenet169, 'densenet'],
                     'densenet201': [densenet201, 'densenet'], 'dpn68': [dpn68, 'dpn832'],
                     'dpn68b': [dpn68b, 'dpn832'], 'dpn92': [dpn92, 'dpn2688'],
                     'dpn98': [dpn98, 'dpn2688'], 'dpn107': [dpn107, 'dpn2688'],
                     'dpn131': [dpn131, 'dpn2688'],
                     'inceptionresnetv2': [inceptionresnetv2, 'inceptionresnetv2'],
                     'inception_v3': [inception_v3, 'inception'], 'resnet50': [resnet50, 'resnet'],
                     'resnet101': [resnet101, 'resnet'], 'resnet152': [resnet152, 'resnet'],
                     'resnet34fpn5': [resnet34fpn5, 'resnet_ssl'],
                     'resnet50fpn5': [resnet50fpn5, 'resnet'],
                     'resnet101fpn5': [resnet101fpn5, 'resnet'],
                     'resnet152fpn5': [resnet152fpn5, 'resnet'],
                     'resnest50': [resnest50, 'resnet'],
                     'resnest101': [resnest101, 'resnet'],
                     'resnest200': [resnest200, 'resnet'],
                     'resnest269': [resnest269, 'resnet'],
                     'gcn_resnest200': [gcn_resnest200, 'resnet'],
                     'gcn_resnest101': [gcn_resnest101, 'resnet'],
                     'gcn_resnest50': [gcn_resnest50, 'resnet'],
                     'xception_dropout': [xception_dropout, 'xception'], 'dla34': [dla34, 'dla'],
                     'bam_dla102x': [bam_dla102x, 'dla'],
                     'dla60x': [dla60x, 'dla'], 'dla102x': [dla102x, 'dla'],
                     'dla169': [dla169, 'dla'], 'dla102x2': [dla102x2, 'dla'],
                     'senet154': [senet154, 'se_resnext'],
                     'se_resnext50': [se_resnext50_32x4d, 'se_resnext'],
                     'se_resnext101': [se_resnext101_32x4d, 'se_resnext'],
                     'resnext50_ssl': [resnext50_32x4d_ssl, 'resnext_ssl'],
                     'resnext50_swsl': [resnext50_32x4d_swsl, 'resnext_ssl'],
                     'resnext101_ssl': [resnext101_32x16d_ssl, 'resnext_ssl'],
                     'resnext101_swsl': [resnext101_32x16d_swsl, 'resnext_ssl'],
                     'resnet18_ssl': [resnet18_ssl, 'resnet_ssl'],
                     'resnet18_swsl': [resnet18_swsl, 'resnet_ssl'],
                     'tf_efficientnet_b0_ns': [tf_efficientnet_b0_ns, 'tf_efficientnet_ns'],
                     'tf_efficientnet_b1_ns': [tf_efficientnet_b1_ns, 'tf_efficientnet_ns'],
                     'tf_efficientnet_b4_ns': [tf_efficientnet_b4_ns, 'tf_efficientnet_ns_b4']}
        return __factory
