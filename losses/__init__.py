from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
from .DiceLoss import DiceLoss
from .F1Loss import F1Loss
from .FocalLoss import FocalLoss
from .BCELogLoss import BCELogLoss
from .BCELogLossV2 import BCELogLossV2
from .FCLoss import BFocalLoss
from .FusedJSD import JsdCrossEntropy, LabelSmoothingCrossEntropy
from .ClassBalancedLoss import CBalancedLoss

__factory = {
    'dice': DiceLoss,
    'margin': nn.MultiLabelSoftMarginLoss,
    'focal': FocalLoss,
    'f1': F1Loss,
    'bce': BCELogLoss,
    'ce': nn.CrossEntropyLoss,
    'bce_v2': BCELogLossV2,
    'bfocal': BFocalLoss,
    'jsd': JsdCrossEntropy,
    'lsce': LabelSmoothingCrossEntropy,
    'class_balance': CBalancedLoss
}


def init_loss_func(name, **kwargs):
    avai_losses = list(__factory.keys())
    if name not in avai_losses:
        raise KeyError(
            'Invalid loss function name. Received "{}", but expected to be one of {}'.format(name, avai_losses))
    return __factory[name](**kwargs)
