import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    This implements Focal Loss following the paper https://arxiv.org/abs/1708.02002
    """

    def __init__(self, device):
        super(FocalLoss, self).__init__()
        self.alpha = 1
        self.gamma = 2
        self.device = device

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        fc_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return torch.mean(fc_loss)
