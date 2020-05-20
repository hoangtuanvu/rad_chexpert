import sys
import math
import torch
import warnings
import torchsummary
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from models.utils import ChannelGate, SpatialGate, get_norm

warnings.filterwarnings("ignore")


class Conv2dNormRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True,
                 norm_type='Unknown'):
        super(Conv2dNormRelu, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
                                  get_norm(norm_type, out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class SAModule(nn.Module):
    """
    Re-implementation of spatial attention module (SAM) described in:
    *Liu et al., Dual Attention Network for Scene Segmentation, cvpr2019
    code reference:
    https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8,
                               kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = self.conv1(feat_map).view(batch_size, -1, width * height).permute(0, 2, 1)

        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = F.softmax(relation_map, dim=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map


class FPAModule(nn.Module):
    """
    Re-implementation of feature pyramid attention (FPA) described in:
    *Li et al., Pyramid Attention Network for Semantic segmentation, Face++2018
    """

    def __init__(self, num_channels, norm_type):
        super(FPAModule, self).__init__()

        # global pooling branch
        self.gap_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        Conv2dNormRelu(num_channels, num_channels, kernel_size=1,
                                                       norm_type=norm_type))

        # middle branch
        self.mid_branch = Conv2dNormRelu(num_channels, num_channels, kernel_size=1,
                                         norm_type=norm_type)

        self.downsample1 = Conv2dNormRelu(num_channels, 1, kernel_size=7, stride=2, padding=3,
                                          norm_type=norm_type)

        self.downsample2 = Conv2dNormRelu(1, 1, kernel_size=5, stride=2, padding=2,
                                          norm_type=norm_type)

        self.downsample3 = Conv2dNormRelu(1, 1, kernel_size=3, stride=2, padding=1,
                                          norm_type=norm_type)

        self.scale1 = Conv2dNormRelu(1, 1, kernel_size=7, padding=3, norm_type=norm_type)
        self.scale2 = Conv2dNormRelu(1, 1, kernel_size=5, padding=2, norm_type=norm_type)
        self.scale3 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1, norm_type=norm_type)

    def forward(self, feat_map):
        height, width = feat_map.size(2), feat_map.size(3)
        gap_branch = self.gap_branch(feat_map)
        gap_branch = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(
            gap_branch)

        mid_branch = self.mid_branch(feat_map)

        scale1 = self.downsample1(feat_map)
        scale2 = self.downsample2(scale1)
        scale3 = self.downsample3(scale2)

        scale3 = self.scale3(scale3)
        scale3 = nn.Upsample(size=(math.ceil(height / 4), math.ceil(width / 4)), mode='bilinear',
                             align_corners=False)(scale3)
        scale2 = self.scale2(scale2) + scale3
        scale2 = nn.Upsample(size=(math.ceil(height / 2), math.ceil(width / 2)), mode='bilinear',
                             align_corners=False)(scale2)
        scale1 = self.scale1(scale1) + scale2
        scale1 = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(scale1)

        feat_map = torch.mul(scale1, mid_branch) + gap_branch

        return feat_map


class SEModule(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BamBlock(nn.Module):
    """
    BAM attention block for BAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """

    def __init__(self, channels):
        super(BamBlock, self).__init__()
        self.ch_att = ChannelGate(channels=channels)
        self.sp_att = SpatialGate(channels=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = 1 + self.sigmoid(self.ch_att(x) * self.sp_att(x))
        x = x * att
        return x


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False,
                 stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size,
                                  stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv,
                                  kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh,
                                               "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels, pool_size=[1, 1], norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels / 4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels), nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels), nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
            norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
            norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels), nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
                                   norm_layer(in_channels))  # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=8):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups,
                           bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", 'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), 'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * h * w) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []

            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class AttentionMap(nn.Module):

    def __init__(self, cfg, num_channels):
        super(AttentionMap, self).__init__()
        self.cfg = cfg

        if self.cfg.attention == "None":
            self.attention = nn.Identity()
        elif self.cfg.attention_map == "SE":
            self.attention = SEModule(num_channels)
        elif self.cfg.attention_map == "SAM":
            self.attention = SAModule(num_channels)
        elif self.cfg.attention_map == "FPA":
            self.attention = FPAModule(num_channels, cfg.norm_type)
        elif self.cfg.attention_map == "STRIP":
            self.attention = StripPooling(num_channels, cfg.norm_type)
        else:
            Exception('Unknown attention type : {}'.format(self.cfg.attention))

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map):
        return self.attention(feat_map)


class WildcatPool2dFunction(Function):
    def __init__(self, kmax, kmin, alpha):
        super(WildcatPool2dFunction, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True,
                   out=(sorted, indices))

        self.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0 and self.alpha is not 0:
            self.indices_min = indices.narrow(2, n - kmin, kmin)
            output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).mul_(self.alpha / kmin)).div_(2)

        self.save_for_backward(input)
        return output.view(batch_size, num_channels)

    def backward(self, grad_output):

        input, = self.saved_tensors

        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size,
                                                                               num_channels, kmax)

        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2,
                                                                                              self.indices_max,
                                                                                              grad_output_max).div_(
            kmax)

        if kmin > 0 and self.alpha is not 0:
            grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size,
                                                                                   num_channels,
                                                                                   kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(
                0).scatter_(2, self.indices_min, grad_output_min).mul_(self.alpha / kmin)
            grad_input.add_(grad_input_min).div_(2)

        return grad_input.view(batch_size, num_channels, h, w)


class WildcatPool2d(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=1):
        super(WildcatPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction(self.kmax, self.kmin, self.alpha)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(
            self.kmin) + ', alpha=' + str(self.alpha) + ')'


class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print(
                'Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    def backward(self, grad_output):
        input, = self.saved_tensors

        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size,
                                                                               num_outputs,
                                                                               self.num_maps, h,
                                                                               w).contiguous()

        return grad_input.view(batch_size, num_channels, h, w)


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)


if __name__ == "__main__":
    att = SpatialCGNL(2048, 1024)
    print(torchsummary.summary(att, input_size=(2048, 7, 7), device='cpu'))
    x = torch.randn([3, 2048, 7, 7])
    y = att(x)
    print(y.size())
