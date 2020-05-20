# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class BCELogLoss(nn.Module):
    """
    Binary Cross-Entropy with positive weights to control imbalance data
    """

    def __init__(self, device):
        super(BCELogLoss, self).__init__()
        self.device = device

    def forward(self, output, target):
        weights = list()
        for index in range(target.size()[1]):
            _target = target[:, index].view(-1)
            if _target.sum() == 0:
                weights.append(0.0)
            else:
                weight = (_target.size()[0] - _target.sum()) / _target.sum()
                weights.append(weight)

        weights = torch.FloatTensor(weights).to(self.device)
        loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=weights)

        dice = dice_loss(torch.sigmoid(output), target)

        _loss = loss + dice
        return _loss
