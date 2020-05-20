from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target, index):
        # output = torch.sigmoid(output)
        print(output, target)
        output = output[index].view(-1)
        target = target[:,index].view(-1)
        if target.sum() == 0:
            output = 1.0 - output
            target = 1.0 - target

        num = (2 * (output * target).sum() + self.smooth)
        den = (output.sum() + target.sum() + self.smooth + self.eps)
        return 1.0 -  num/den 
