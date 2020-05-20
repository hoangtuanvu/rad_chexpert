import torch
import torch.nn as nn


class BFocalLoss(nn.Module):
    def __init__(self, device, alpha=1, gamma=2, logits=False, reduce=True):
        super(BFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.device = device

    def forward(self, output, target):
        output = torch.sigmoid(output)

        if target.size(1) == 6:
            weights = torch.Tensor([[0.732229646,  # 0.960300001,
                                     # 0.921019512,
                                     # 0.883857039,
                                     0.691485987, 0.943847012,  # 0.849559514,
                                     0.904791626, 0.528020218,  # 0.981336908,
                                     # 0.997922912,
                                     # 0.933318326,
                                     # 0.939263094,
                                     # 0.992254407,
                                     # 0.962939846,
                                     # 0.62530568,
                                     # 0.99958049,
                                     0.888389797,  # 0.757727687
                                     ]]).to(self.device)
        else:
            weights = torch.Tensor([[0.732229646, 0.960300001, 0.921019512, 0.883857039,
                                     0.691485987, 0.943847012, 0.849559514, 0.904791626,
                                     0.528020218, 0.981336908, 0.997922912, 0.933318326,
                                     0.939263094, 0.992254407, 0.962939846, 0.62530568, 0.99958049,
                                     0.888389797, 0.757727687]]).to(self.device)

        m = target.size()[0]
        nnClasses = target.size()[1]
        epsilon = 1e-10

        pos_wt = (torch.ones((m, nnClasses), dtype=torch.float)).to(self.device) * weights.float()
        neg_wt = 1 - pos_wt
        wt = target * pos_wt + (1 - target) * neg_wt

        unweighted_loss = - (target * (torch.log(output + epsilon)) + (1 - target) * (
            torch.log(1 - output + epsilon)))
        weighted_loss = unweighted_loss * wt

        loss = (weighted_loss.sum()) / m

        return loss