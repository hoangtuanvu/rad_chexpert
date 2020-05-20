import torch.nn as nn
import torch.nn.functional as F


class BCELogLossV2(nn.Module):
    """
    Binary Cross-Entropy with positive weights to control imbalance data
    """

    def __init__(self, device):
        super(BCELogLossV2, self).__init__()
        self.device = device

    def forward(self, output, target):
        loss = 0
        for index in range(target.size()[1]):
            _target = target[:, index].view(-1)
            if _target.sum() == 0:
                _loss = 0.0
            else:
                weight = (_target.size()[0] - _target.sum()) / _target.sum()
                _loss = F.binary_cross_entropy_with_logits(
                    output[:, index].view(-1), _target, pos_weight=weight)

            loss += _loss

        return loss
