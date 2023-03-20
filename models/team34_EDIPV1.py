import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
def make_model(args, parent=False):
    model = EDIPv1()
    return model


class Conv2D_WN(nn.Conv2d):
    '''Conv2D with weight normalization.
    '''
    def __init__(self, 
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        super(Conv2D_WN, self).__init__(in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, 
            bias=bias, padding_mode=padding_mode)

        # set up the scale variable in weight normalization
        self.weight_g = nn.Parameter(torch.ones(out_channels), requires_grad=True)
        self.init_wn()
    
    def init_wn(self):
        """initialize the wn parameters"""
        for i in range(self.weight.size(0)):
            self.weight_g.data[i] = torch.norm(self.weight.data[i])

    def forward(self, input):
        w = F.normalize(self.weight, dim=(1,2,3))
        w = w * self.weight_g.view(-1,1,1,1)
        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, wn=False):
    padding = int((kernel_size - 1) / 2) * dilation
    if wn:
        return Conv2D_WN(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                         groups=groups)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                         groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', wn=True):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0
    if wn:
        c = Conv2D_WN(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=groups)
    else:
        c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_indices(m):
    wg = m.weight_g.data.squeeze()

    # print(wg)
    # print(wg)
    # input()
    _, indices = torch.sort(wg)
    pruned_n = int(wg.shape[0] * 0.3125)
    thre = wg[indices[pruned_n]]
    indice = torch.range(0, wg.shape[0] - 1).to(torch.long)
    mask = wg >= thre
    # input()
    return indice[mask]
    #return indice


def build_new(m, pre, post):
    t = conv_layer(len(pre), len(post), m.weight.data.shape[3], wn=False)
    # print("-"*100)
    # print(m.weight.data.shape, pre, post)
    # input()
    kept_weights = m.weight.data[post][:, pre, :, :]
    t.weight.data.copy_(kept_weights)
    kept_bias = m.bias.data[post]
    t.bias.data.copy_(kept_bias)
    return t


def build_new_block(m, pre, post):
    t = conv_block(len(pre), len(post), kernel_size=1, act_type='lrelu', wn=False)
    kept_weights = m[0].weight.data[post][:, pre, :, :]
    t[0].weight.data.copy_(kept_weights)
    kept_bias = m[0].bias.data[post]
    t[0].bias.data.copy_(kept_bias)
    return t


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class RLFB_Rep_Prune(nn.Module):
    def __init__(self, in_channels, esa_channels=16,mid_channels=31):
        super(RLFB_Rep_Prune, self).__init__()

        self.c1_r = conv_layer(in_channels, mid_channels, 3,bias=False)

        # self.c1_r_11_m = conv_layer(256, in_channels, 1,bias=False)
        # self.c1_r_11_a = conv_layer(in_channels, in_channels, 1,bias=False)



        self.c2_r = conv_layer(mid_channels, mid_channels, 3,bias=False)
        # self.c2_r_11_m = conv_layer(256, in_channels, 1,bias=False)
        # self.c2_r_11_a = conv_layer(in_channels, in_channels, 1,bias=False)


        self.c3_r = conv_layer(mid_channels, in_channels, 3,bias=False)
        # self.c3_r_11_m = conv_layer(256, in_channels, 1,bias=False)
        # self.c3_r_11_a = conv_layer(in_channels, in_channels, 1,bias=False)

        self.c5 = conv_layer(in_channels, 44, 1,bias=False)
        self.esa = ESA(esa_channels, 44, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out_33 = (self.c1_r(x))
        out = self.act(out_33)

        out_33 = (self.c2_r(out))
        out = self.act(out_33)

        out_33 = (self.c3_r(out))
        out = self.act(out_33)

        out = out + x
        out = self.esa(self.c5(out))

        return out

    def reparam(self):
        oc, ic = self.c1_r_11_m.weight.data.shape[0], self.c1_r.weight.data.shape[1]
        kernel_value = F.conv2d(self.c1_r.weight.data.permute(1, 0, 2, 3),
                                self.c1_r_11_m.weight.data).permute(1, 0, 2, 3)
        for i in range(oc):
            kernel_value[i, i % ic, 1, 1] += 1

        for i in range(oc):
            for j in range(ic):
                kernel_value[i, j, 1, 1] += (self.c1_r_11_a.weight.data[i, j, :, :]).numpy()
        # rep时使用
        # self.c1_r = conv_layer(ic, oc, 3,)
        self.c1_r.weight.data = kernel_value
        self.c1_r.to(self.c1_r_11_m.weight.device)

        oc, ic = self.c2_r_11_m.weight.data.shape[0], self.c2_r.weight.data.shape[1]
        kernel_value = F.conv2d(self.c2_r.weight.data.permute(1, 0, 2, 3),
                                self.c2_r_11_m.weight.data).permute(1, 0, 2, 3)
        for i in range(oc):
            kernel_value[i, i % ic, 1, 1] += 1
        for i in range(oc):
            for j in range(ic):
                kernel_value[i, j, 1, 1] += (self.c2_r_11_a.weight.data[i, j, :, :]).numpy()
        # rep时使用
        # self.c2_r = conv_layer(ic, oc, 3)
        self.c2_r.weight.data = kernel_value
        self.c2_r.to(self.c2_r_11_m.weight.device)

        oc, ic = self.c3_r_11_m.weight.data.shape[0], self.c3_r.weight.data.shape[1]
        kernel_value = F.conv2d(self.c3_r.weight.data.permute(1, 0, 2, 3),
                                self.c3_r_11_m.weight.data).permute(1, 0, 2, 3)
        for i in range(oc):
            kernel_value[i, i % ic, 1, 1] += 1
        for i in range(oc):
            for j in range(ic):
                kernel_value[i, j, 1, 1] += (self.c3_r_11_a.weight.data[i, j, :, :]).numpy()
        # rep时使用
        # self.c3_r = conv_layer(ic, oc, 3)
        self.c3_r.weight.data = kernel_value
        self.c3_r.to(self.c3_r_11_a.weight.device)

        self.forward = self.forwar_rep

    def forwar_rep(self, x):
        out_33 = (self.c1_r(x))
        out = self.act(out_33)

        out_33 = (self.c2_r(out))
        out = self.act(out_33)

        out_33 = (self.c3_r(out))
        out = self.act(out_33)

        out = out + x
        out = self.esa(self.c5(out))

        return out

    def apply_wn(self):
        alist = [self.c1_r, self.c1_r_11, self.c2_r, self.c2_r_11, self.c3_r, self.c3_r_11, self.c5]
        for m in alist:
            m.apply(nn.utils.weight_norm)

    def get_gammas(self):
        gammas = []
        for m in self.modules():
            if hasattr(m, 'weight_g'):
                gammas.append(m.weight_g.squeeze())
        return gammas

    def prunef(self, pre_channels):

        pre_channel = pre_channels
        c1r_indice1 = get_indices(self.c1_r)
        self.c1_r = build_new(self.c1_r, pre_channel, c1r_indice1)

        pre_channel = torch.split(c1r_indice1, 1)
        c2r_indice1 = get_indices(self.c2_r)
        self.c2_r = build_new(self.c2_r, pre_channel, c2r_indice1)

        pre_channel = torch.split(c2r_indice1, 1)
        c3r_indice1 = get_indices(self.c3_r)
        self.c3_r = build_new(self.c3_r, pre_channel, c3r_indice1)

        pre_channel = torch.split(c3r_indice1, 1)
        c_indice5 = get_indices(self.c5)
        self.c5 = build_new(self.c5, pre_channel, c_indice5)

        self.esa.conv1 = build_new(self.esa.conv1, torch.split(c_indice5, 1),
                                   torch.tensor(range(self.esa.conv1.weight.data.shape[0])))
        self.esa.conv4 = build_new(self.esa.conv4, torch.tensor(range(self.esa.conv4.weight.data.shape[1])), c_indice5)
        return c_indice5



class EDIPv1(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=44,
                 upscale=4):
        super(EDIPv1, self).__init__()

        self.conv_1 = conv_layer(3, feature_channels, kernel_size=3)

        self.block_1 = RLFB_Rep_Prune(in_channels=44,mid_channels=32)
        self.block_2 = RLFB_Rep_Prune(in_channels=44,mid_channels=32)
        self.block_3 = RLFB_Rep_Prune(in_channels=44,mid_channels=32)
        self.block_4 = RLFB_Rep_Prune(in_channels=44,mid_channels=32)
        # self.block_5 = RLFB_Rep(feature_channels)
        # self.block_6 = RLFB_Rep(feature_channels)

        self.conv_2 = conv_layer(44,
                                   feature_channels,
                                   kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                          out_channels,
                                          upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        # out_b5 = self.block_5(out_b4)
        # out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)
        # output = torch.clamp(output,0.0,1.0)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def rep(self):
        for m in self.modules():
            if hasattr(m, 'reparam'):
                m.reparam()

    def apply_weight_norm(self):
        self.conv_1.apply(nn.utils.weight_norm)
        self.conv_2.apply(nn.utils.weight_norm)
        for m in self.modules():
            if hasattr(m, 'apply_wn'):
                m.apply_wn()

    def prune(self):
        for m in self.modules():
            if hasattr(m, 'weight_g'):
                m.weight_g.data = m.weight_g.data.view(-1, 1, 1, 1)
                m.weight.data = F.normalize(m.weight.data, dim=(1, 2, 3)) * m.weight_g

        indice = get_indices(self.conv_1)
        self.conv_1 = build_new(self.conv_1, [0, 1, 2], indice)
        pre_channel = indice
        pre_channels = []
        for m in self.modules():
            if hasattr(m, 'prunef'):
                pre_channel = m.prunef(pre_channel)
                pre_channels.append(pre_channel)
        # feat_indice = []
        # offset = 0
        # for p in pre_channels:
        #     for i in p:
        #         feat_indice.append(i + offset)
        #     offset += self.nf

        # selfc_indice = get_indices(self.c[0])
        # self.c = build_new_block(self.c, feat_indice, selfc_indice)

        conv2_indice = get_indices(self.conv_2)
        self.conv_2 = build_new(self.conv_2, torch.split(pre_channels[-1], 1), indice)

        self.upsampler[0] = build_new(self.upsampler[0], torch.split(indice, 1), torch.tensor(range(3 * (4 ** 2))))

    def get_all_gammas(self):
        gammas = [self.fea_conv.weight_g.squeeze(), self.LR_conv.weight_g.squeeze(), self.c[0].weight_g.squeeze()]
        for m in self.modules():
            if hasattr(m, 'get_gammas'):
                gammas += m.get_gammas()

        return gammas

    def init_all_wn(self):
        for m in self.modules():
            if hasattr(m, 'init_wn'):
                m.init_wn()



