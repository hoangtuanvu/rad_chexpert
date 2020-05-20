"""Summary
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from models.common import conv1x1, conv1x1_block, conv3x3_block
from torch.nn.parameter import Parameter
import pandas as pd
import math


def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    """Summary
    
    Args:
        norm_type (TYPE): Description
        num_features (TYPE): Description
        num_groups (int, optional): Description
        eps (float, optional): Description
    
    Returns:
        TYPE: Description
    
    Raises:
        Exception: Description
    """
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps, affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))


class LinearScheduler(nn.Module):
    """Summary
    
    Attributes:
        drop_values (TYPE): Description
        dropblock (TYPE): Description
        i (int): Description
    """

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        """Summary
        
        Args:
            dropblock (TYPE): Description
            start_value (TYPE): Description
            stop_value (TYPE): Description
            nr_steps (TYPE): Description
        """
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return self.dropblock(x)

    def step(self):
        """Summary
        """
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    
    Shape:
            - Input: `(N, C, H, W)`
            - Output: `(N, C, H, W)`
    
        .. _DropBlock: A regularization method for convolutional networks:
           https://arxiv.org/abs/1810.12890
    
    Attributes:
        block_size (TYPE): Description
        drop_prob (TYPE): Description
    
    """

    def __init__(self, drop_prob, block_size):
        """Summary
        
        Args:
            drop_prob (TYPE): Description
            block_size (TYPE): Description
        """
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        """Summary
        
        Args:
            mask (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size), stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return self.drop_prob / (self.block_size ** 2)


class ChannelGate(nn.Module):
    """
    BAM channel gate block.
    
    Parameters:
        ----------
        channels: int
            Number of input/output channels.
        reduction_ratio: int, default 16
            Channel reduction ratio.
        num_layers: int, default 1
            Number of dense blocks.
    
    Attributes:
        final_fc (TYPE): Description
        init_fc (TYPE): Description
        main_fcs (TYPE): Description
        pool (TYPE): Description
    """

    def __init__(self, channels, reduction_ratio=16, num_layers=1):
        """Summary
        
        Args:
            channels (TYPE): Description
            reduction_ratio (int, optional): Description
            num_layers (int, optional): Description
        """
        super(ChannelGate, self).__init__()
        mid_channels = channels // reduction_ratio

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.init_fc = DenseBlock(in_features=channels, out_features=mid_channels)
        self.main_fcs = nn.Sequential()
        for i in range(num_layers - 1):
            self.main_fcs.add_module("fc{}".format(i + 1), DenseBlock(in_features=mid_channels,
                                                                      out_features=mid_channels))
        self.final_fc = nn.Linear(in_features=mid_channels, out_features=channels)

    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        input = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.init_fc(x)
        x = self.main_fcs(x)
        x = self.final_fc(x)
        x = x.unsqueeze(2).unsqueeze(3).expand_as(input)
        return x


class SpatialGate(nn.Module):
    """
    BAM spatial gate block.
    
    Parameters:
        ----------
        channels: int
            Number of input/output channels.
        reduction_ratio: int, default 16
            Channel reduction ratio.
        num_dil_convs: int, default 2
            Number of dilated convolutions.
        dilation: int, default 4
            Dilation/padding value for corresponding convolutions.
    
    Attributes:
        dil_convs (TYPE): Description
        final_conv (TYPE): Description
        init_conv (TYPE): Description
    """

    def __init__(self, channels, reduction_ratio=16, num_dil_convs=2, dilation=4):
        """Summary
        
        Args:
            channels (TYPE): Description
            reduction_ratio (int, optional): Description
            num_dil_convs (int, optional): Description
            dilation (int, optional): Description
        """
        super(SpatialGate, self).__init__()
        mid_channels = channels // reduction_ratio

        self.init_conv = conv1x1_block(in_channels=channels, out_channels=mid_channels, stride=1,
                                       bias=True)
        self.dil_convs = nn.Sequential()
        for i in range(num_dil_convs):
            self.dil_convs.add_module("conv{}".format(i + 1),
                                      conv3x3_block(in_channels=mid_channels,
                                                    out_channels=mid_channels, stride=1,
                                                    padding=dilation, dilation=dilation, bias=True))
        self.final_conv = conv1x1(in_channels=mid_channels, out_channels=1, stride=1, bias=True)

    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        input = x
        x = self.init_conv(x)
        x = self.dil_convs(x)
        x = self.final_conv(x)
        x = x.expand_as(input)
        return x


class DenseBlock(nn.Module):
    """
    Standard dense block with Batch normalization and ReLU activation.
    
    Parameters:
        ----------
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.
    
    Attributes:
        activ (TYPE): Description
        bn (TYPE): Description
        fc (TYPE): Description
    """

    def __init__(self, in_features, out_features):
        """Summary
        
        Args:
            in_features (TYPE): Description
            out_features (TYPE): Description
        """
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.bn = nn.BatchNorm1d(num_features=out_features)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        x = self.fc(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def _calc_width(net):
    """Summary
    
    Args:
        net (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def load_state_dict(model_path, model, device):
    """Summary
    
    Args:
        model_path (TYPE): Description
        model (TYPE): Description
        device (TYPE): Description
    """

    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()

    for k, v in state_dict['state_dict'].items():
        if k.startswith('module'):
            name = k[7:]
        else:
            name = k

        if k.startswith('model'):
            name = name[6:]

        new_state_dict[name] = v

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    return state_dict


def beta_mish(input, beta=-0.25):
    '''
    Applies the β mish function element-wise:
    β mish(x) = x * tanh(ln((1 + exp(x))^β))
    See additional documentation for beta_mish class.
    '''
    return input * torch.tanh(torch.log(torch.pow((1 + torch.exp(input)), beta)))


class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 buffer_num=0, rho=1.0, burnin=0, two_stage=True, FROZEN=False, out_p=False):
        super(CBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.buffer_num = buffer_num
        self.max_buffer_num = buffer_num
        self.rho = rho
        self.burnin = burnin
        self.two_stage = two_stage
        self.FROZEN = FROZEN
        self.out_p = out_p

        self.iter_count = 0
        self.pre_mu = []
        self.pre_meanx2 = []  # mean(x^2)
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.ones = torch.ones(self.num_features).cuda()

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))

    def forward(self, input, weight):
        # deal with wight and grad of self.pre_dxdw!
        self._check_input_dim(input)
        y = input.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(input.size(1), -1)

        # burnin
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and input.requires_grad and len(
                self.pre_mu) == self.buffer_num:  # some layers are frozen!
            # cal current batch mu and sigma
            cur_mu = y.mean(dim=1)
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            cur_sigma2 = y.var(dim=1)
            # cal dmu/dw dsigma2/dw
            dmudw = torch.autograd.grad(cur_mu, weight, self.ones, retain_graph=True)[0]
            dmeanx2dw = torch.autograd.grad(cur_meanx2, weight, self.ones, retain_graph=True)[0]
            # update cur_mu and cur_sigma2 with pres
            mu_all = torch.stack([cur_mu, ] + [
                tmp_mu + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for
                tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)])
            meanx2_all = torch.stack([cur_meanx2, ] + [
                tmp_meanx2 + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for
                tmp_meanx2, tmp_d, tmp_w in
                zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)])
            sigma2_all = meanx2_all - torch.pow(mu_all, 2)

            # with considering count
            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()
            re_mu_all[sigma2_all < 0] = 0
            re_meanx2_all[sigma2_all < 0] = 0
            count = (sigma2_all >= 0).sum(dim=0).float()
            mu = re_mu_all.sum(dim=0) / count
            sigma2 = re_meanx2_all.sum(dim=0) / count - torch.pow(mu, 2)

            self.pre_mu = [cur_mu.detach(), ] + self.pre_mu[:(self.buffer_num - 1)]
            self.pre_meanx2 = [cur_meanx2.detach(), ] + self.pre_meanx2[:(self.buffer_num - 1)]
            self.pre_dmudw = [dmudw.detach(), ] + self.pre_dmudw[:(self.buffer_num - 1)]
            self.pre_dmeanx2dw = [dmeanx2dw.detach(), ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = torch.zeros_like(weight.data)
            tmp_weight.copy_(weight.data)
            self.pre_weight = [tmp_weight.detach(), ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            x = y
            mu = x.mean(dim=1)
            cur_mu = mu
            sigma2 = x.var(dim=1)
            cur_sigma2 = sigma2

        if not self.training or self.FROZEN:
            y = y - self.running_mean.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (self.running_var.view(-1, 1) + self.eps) ** .5
            else:
                y = y / (self.running_var.view(-1, 1) ** .5 + self.eps)

        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (
                                                1 - self.momentum) * self.running_mean + self.momentum * cur_mu
                    self.running_var = (
                                               1 - self.momentum) * self.running_var + self.momentum * cur_sigma2
            y = y - mu.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (sigma2.view(-1, 1) + self.eps) ** .5
            else:
                y = y / (sigma2.view(-1, 1) ** .5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'buffer={max_buffer_num}, burnin={burnin}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


def gen_adj_num(labels, agree_rate=0.5, csv_path='', t=0.4):
    df_valid = pd.read_csv(csv_path)
    df_valid = df_valid.fillna(0)
    num_classes = len(labels.split(','))
    indices = list(map(int, labels.split(',')))
    label_names = [list(df_valid)[idx + 1] for idx in indices]
    df_valid = df_valid[label_names]

    if agree_rate > 0:
        df_valid[list(df_valid)] = 1 * (df_valid[list(df_valid)] >= agree_rate)

    data_matric = df_valid.iloc[:].copy().values
    data_matric = np.array(data_matric, dtype=np.long)
    data_matric = np.abs(data_matric)
    _number = []
    datasize, number_labels = data_matric.shape
    for i in range(number_labels):
        nozero = np.count_nonzero(data_matric[:, i])
        _number.append(nozero)
    _nums = np.array(_number)
    _nums = _nums[:, np.newaxis]
    correlation_matrix = np.zeros((number_labels, number_labels), dtype=int)
    for i in range(number_labels):
        for j in range(i + 1, number_labels):
            a = data_matric[:, i]
            b = data_matric[:, j]
            c = a & b
            num = np.count_nonzero(c)
            correlation_matrix[i][j] = num
            correlation_matrix[j][i] = num
    _adj = correlation_matrix
    _adj = _adj / _nums
    # _adj[_adj < t] = 0
    # _adj[_adj >= t] = 1
    # _adj = _adj * 0.25 / (_adj.sum(0) + 1e-6)
    # _adj = _adj + np.identity(number_labels, np.int)

    # D = _adj.sum(1)
    # half_D = D ** (-0.5)
    # half_D = np.diag(half_D)
    # _adj = np.matmul(np.matmul(half_D, _adj), half_D)
    # _adj[_adj < 0.05] = 0
    # neighbor_rate = 0.4
    # _adj = neighbor_rate * _adj + (1 - neighbor_rate) * np.eye(num_classes)

    _adj[_adj < 0.1] = 0
    neighbor_rate = 0.4
    _adj = neighbor_rate * _adj + (1 - neighbor_rate) * np.eye(num_classes)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def input_embedding(num_classes):
    inp = np.identity(num_classes)
    inp = torch.from_numpy(inp).float()
    inp = torch.unsqueeze(inp, 0)
    return inp


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, supp_features=0, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.supp_features = supp_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)

        if self.supp_features:
            print(self.supp_features)
            self.supp_w = Parameter(torch.Tensor(supp_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        if self.supp_features:
            supp_stdv = 1. / math.sqrt(self.supp_w.size(1))
            self.supp_w.data.uniform_(-supp_stdv, supp_stdv)

    def forward(self, input, adj):
        if self.supp_features > 0:
            input = torch.matmul(self.supp_w, input)

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(
            self.out_features) + ')'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
