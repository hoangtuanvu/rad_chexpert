import torch
import torch.nn.functional as F
from torch import nn


class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        (N, C, H, W) = feat_map.shape

        # (N, C, 1, 1) m
        m, _ = torch.max(feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        # TODO: split dim=(-1, -2) for onnx.export
        return m + 1 / g * torch.log(
            area * torch.sum(torch.exp(g * value0), dim=(-1, -2), keepdim=True))


class MixedPool(nn.Module):
    def __init__(self, a):
        super(MixedPool, self).__init__()
        self.a = nn.Parameter(a * torch.ones(1))

    def forward(self, x):
        return self.a * F.adaptive_max_pool2d(x, 1) + (1 - self.a) * F.adaptive_avg_pool2d(x, 1)


class LSEPool(nn.Module):
    """
    Learnable LSE pooling with a shared parameter
    """

    def __init__(self, r):
        super(LSEPool, self).__init__()
        self.r = nn.Parameter(torch.ones(1) * r)

    def forward(self, x):
        s = (x.size(2) * x.size(3))
        x_max = F.adaptive_max_pool2d(x, 1)
        exp = torch.exp(self.r * (x - x_max))
        sumexp = 1 / s * torch.sum(exp, dim=(2, 3))
        sumexp = sumexp.view(sumexp.size(0), -1, 1, 1)
        logsumexp = x_max + 1 / self.r * torch.log(sumexp)
        return logsumexp


class ExpPool(nn.Module):

    def __init__(self):
        super(ExpPool, self).__init__()

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """

        EPSILON = 1e-7
        (N, C, H, W) = feat_map.shape
        m, _ = torch.max(feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # caculate the sum of exp(xi)
        # TODO: split dim=(-1, -2) for onnx.export
        sum_exp = torch.sum(torch.exp(feat_map - m), dim=(-1, -2), keepdim=True)

        # prevent from dividing by zero
        sum_exp += EPSILON

        # caculate softmax in shape of (H,W)
        exp_weight = torch.exp(feat_map - m) / sum_exp
        weighted_value = feat_map * exp_weight

        # TODO: split dim=(-1, -2) for onnx.export
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


class LinearPool(nn.Module):

    def __init__(self):
        super(LinearPool, self).__init__()

    def forward(self, feat_map):
        """
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        EPSILON = 1e-7
        (N, C, H, W) = feat_map.shape

        # sum feat_map's last two dimention into a scalar
        # so the shape of sum_input is (N,C,1,1)
        # TODO: split dim=(-1, -2) for onnx.export
        sum_input = torch.sum(feat_map, dim=(-1, -2), keepdim=True)

        # prevent from dividing by zero
        sum_input += EPSILON

        # caculate softmax in shape of (H,W)
        linear_weight = feat_map / sum_input
        weighted_value = feat_map * linear_weight

        # TODO: split dim=(-1, -2) for onnx.export
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


class GMP(nn.Module):
    """ Generalized Max Pooling
    """

    def __init__(self, lamb):
        super().__init__()
        self.lamb = nn.Parameter(lamb * torch.ones(1))  # self.inv_lamb = nn.Parameter((1./lamb) * torch.ones(1))

    def forward(self, x):
        B, D, H, W = x.shape
        N = H * W
        identity = torch.eye(N).cuda()
        # reshape x, s.t. we can use the gmp formulation as a global pooling operation
        x = x.view(B, D, N)
        x = x.permute(0, 2, 1)
        # compute the linear kernel
        K = torch.bmm(x, x.permute(0, 2, 1))
        # solve the linear system (K + lambda * I) * alpha = ones
        A = K + self.lamb * identity
        o = torch.ones(B, N, 1).cuda()
        # alphas, _ = torch.gesv(o, A) # tested using pytorch 1.0.1
        alphas, _ = torch.solve(o, A)  # tested using pytorch 1.2.0
        alphas = alphas.view(B, 1, -1)
        xi = torch.bmm(alphas, x)
        xi = torch.transpose(xi, 1, 2).unsqueeze(-1)
        return xi


class GlobalPool(nn.Module):

    def __init__(self, cfg):
        super(GlobalPool, self).__init__()
        self.cfg = cfg
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.exp_pool = ExpPool()
        self.linear_pool = LinearPool()
        self.lse_pool = LSEPool(cfg.lse_gamma)
        self.mix_pool = MixedPool(0.5)
        self.gmp_pool = GMP(cfg.lamb)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map):
        if self.cfg.global_pool == 'AVG':
            return self.avgpool(feat_map)
        elif self.cfg.global_pool == 'MAX':
            return self.maxpool(feat_map)
        elif self.cfg.global_pool == 'AVG_MAX':
            a = self.avgpool(feat_map)
            b = self.maxpool(feat_map)
            return torch.cat((a, b), 1)
        elif self.cfg.global_pool == 'AVG_MAX_LSE':
            a = self.avgpool(feat_map)
            b = self.maxpool(feat_map)
            c = self.lse_pool(feat_map)
            return torch.cat((a, b, c), 1)
        elif self.cfg.global_pool == 'EXP':
            return self.exp_pool(feat_map)
        elif self.cfg.global_pool == 'LINEAR':
            return self.linear_pool(feat_map)
        elif self.cfg.global_pool == 'LSE':
            return self.lse_pool(feat_map)
        elif self.cfg.global_pool == 'MEAN_AVG_MAX':
            a = self.avgpool(feat_map)
            b = self.maxpool(feat_map)
            return (a + b) / 2
        elif self.cfg.global_pool == 'MIXED_POOL':
            return self.mix_pool(feat_map)
        elif self.cfg.global_pool == 'GMP':
            return self.gmp_pool(feat_map)
        else:
            return feat_map


if __name__ == "__main__":
    # avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    # max_pool = nn.AdaptiveMaxPool2d((1, 1))
    # gmp = GMP(lamb=1e3)
    mixed_pool = MixedPool(0.5)
    x = torch.randn([2, 2048, 7, 7])
    # a = avg_pool(x)
    # b = max_pool(x)
    c = mixed_pool(x)

    # c = (a + b) / 2
    print(c.size())
