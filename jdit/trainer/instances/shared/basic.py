# coding=utf-8
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d, ModuleList, Parameter, Softmax, LayerNorm, GroupNorm, ConvTranspose2d
import torch
from torch.nn.utils import spectral_norm


class downsampleBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), pool_type="Max", active_type="ReLU",
                 norm_type=None, groups=1, use_sn=False):
        super(downsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        c_in, c_out = channel_in_out

        self.convLayer_1 = convLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type, groups,
                                     use_sn=use_sn)

        self.poolLayer = getPoolLayer(pool_type)

        # self.convLayer_2 = convLayer(c_out, c_out, kernel_size, stride, padding, active_type, norm_type, groups)

    def forward(self, input):
        out = self.convLayer_1(input)
        if self.poolLayer is not None:
            out = self.poolLayer(out)
        # out = self.convLayer_2(out)
        return out


class upsampleBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None, groups=1, use_sn=False):
        super(upsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        c_in, c_out = channel_in_out
        # c_shrunk = c_in // 2

        # self.shrunkConvLayer_1 = convLayer(c_in, c_shrunk, 1, 1, 0, active_type, norm_type, 1)

        self.convLayer_2 = deconvLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type, groups,
                                       use_sn=use_sn)

        self.upSampleLayer = Upsample(scale_factor=2)

    def forward(self, prev_input, now_input):
        out = torch.cat((prev_input, now_input), 1)
        # out = self.shrunkConvLayer_1(out)
        out = self.upSampleLayer(out)
        out = self.convLayer_2(out)
        return out


class convLayer(Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, active_type="ReLU", norm_type=None, groups=1,
                 use_sn=False):
        super(convLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])

        if use_sn:
            self.module_list += [spectral_norm(Conv2d(c_in, c_out, kernel_size, stride, padding, groups=groups))]
        else:
            self.module_list += [Conv2d(c_in, c_out, kernel_size, stride, padding, groups=groups)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class deconvLayer(Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, padding=1, active_type="ReLU", norm_type=None, groups=1,
                 use_sn=False):
        super(deconvLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])
        if use_sn:
            self.module_list += [
                spectral_norm(ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, groups=groups))]
        else:
            self.module_list += [ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, groups=groups)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class normalBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None, ksp_2=None, groups=1,
                 use_sn=True):
        super(normalBlock, self).__init__()
        c_in, c_out = channel_in_out
        kernel_size, stride, padding = ksp

        if ksp_2 is None:
            kernel_size_2, stride_2, padding_2 = ksp
        else:
            kernel_size_2, stride_2, padding_2 = ksp_2

        assert stride + 2 * padding - kernel_size == 0, \
            "kernel_size%s, stride%s, padding%s is not a 'same' group! s+2p-k ==0" % (
                kernel_size, stride, padding)
        assert stride_2 + 2 * padding_2 - kernel_size_2 == 0, \
            "kernel_size%s, stride%s, padding%s is not a 'same' group! s+2p-k ==0" % (
                kernel_size_2, stride_2, padding_2)

        self.convLayer_1 = convLayer(c_in, c_out, kernel_size_2, stride_2, padding_2, active_type, norm_type,
                                     groups=groups, use_sn=use_sn)
        # self.convLayer_2 = convLayer(c_out, c_out, kernel_size, stride, padding, active_type, norm_type, groups=groups)

    def forward(self, input):
        out = input
        out = self.convLayer_1(out)
        # out = self.convLayer_2(out)

        return out


class SwitchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, last_gamma=False,
                 affine=True):
        super(SwitchNorm, self).__init__()
        self.weight = Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1))
        self.mean_weight = Parameter(torch.ones(3))
        self.var_weight = Parameter(torch.ones(3))
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
        var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


# class ThickConv2d(Module):
#     def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(ThickConv2d, self).__init__()
#         self.out_channels_list = ModuleList([])
#         for i in range(out_channels):
#             self.out_channels_list.append(ConvUnit(in_channels, mid_channels, kernel_size, stride,
#                                                    padding, dilation, groups, bias))
#
#     def forward(self, input):
#         results = []
#         for unit in self.out_channels_list:
#             results.append(unit(input))
#         out = torch.cat(results, 1)
#         return out


# class ConvUnit(ModuleList):
#     def __init__(self, in_channels, mid_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(ConvUnit, self).__init__()
#         assert in_channels is not None
#         self.add_module("conv1",
#                         Conv2d(in_channels, mid_channels, kernel_size, stride, padding, dilation, groups, bias))
#         self.add_module("act", LeakyReLU(0.1))
#         self.add_module("conv2", Conv2d(mid_channels, 1, 1, 1, 0, dilation=1, groups=1, bias=True))
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.act(out)
#         out = self.conv2(out)
#         return out

# class DeConvUnit(ModuleList):
#     def __init__(self, in_channels, mid_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(DeConvUnit, self).__init__()
#         assert in_channels is not None
#         self.add_module("conv1",
#                         ConvTranspose2d(in_channels, mid_channels, kernel_size, stride, padding, dilation=dilation,
#                                         groups=groups, bias=bias))
#         self.add_module("act", LeakyReLU(0.1))
#         self.add_module("conv2", Conv2d(mid_channels, 1, 1, 1, 0, dilation=1, groups=1, bias=True))
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.act(out)
#         out = self.conv2(out)
#         return out

class NLConv2d(Module):
    def __init__(self, in_channels, mid_channels, out_channels, knl_list, strd_list,
                 pad_list, dilation_list, groups_list, bias_list=True):
        super(NLConv2d, self).__init__()
        self.out_channels_list = ModuleList([])

        for i in range(out_channels):
            kernel_size = self.getKernelSuparams(knl_list, i)
            stride = self.getKernelSuparams(strd_list, i)
            padding = self.getKernelSuparams(pad_list, i)
            dilation = self.getKernelSuparams(dilation_list, i)
            groups = self.getKernelSuparams(groups_list, i)
            bias = self.getKernelSuparams(bias_list, i)
            self.out_channels_list.append(
                self.build_convunit(in_channels=in_channels, mid_channels=mid_channels, kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding, dilation=dilation, groups=groups, bias=bias))

    def build_convunit(self, in_channels, mid_channels, kernel_size, stride=1,
                       padding=0, dilation=1, groups=1, bias=True):
        module_list = ModuleList([])
        module_list.append(Conv2d(in_channels, mid_channels, kernel_size, stride, padding, dilation, groups, bias))
        module_list.append(LeakyReLU(0.1))
        module_list.append(Conv2d(mid_channels, 1, 1, 1, 0, dilation=1, groups=1, bias=True))
        ConvUnit = Sequential(*module_list)
        return ConvUnit

    def getKernelSuparams(self, params, index):
        """
        list or tuple and length must grater than 1.
        :param params:
        :param index:
        :return:
        """
        is_list = (isinstance(params, list) or isinstance(params, tuple)) and (len(params) > 1)
        # is_scalar = isinstance(params, int) or isinstance(params, bool) or (is_list and len(params)==1)
        if not is_list:
            param = params
        else:
            param = params[index]
        # else:
        #     raise NameError('params can only be `int` ,`list` or `tuple`. but %s were given. ' % type(params))
        return param

    def forward(self, input):
        results = []
        for unit in self.out_channels_list:
            results.append(unit(input))
        out = torch.cat(results, 1)
        return out


class NLDeConv2d(Module):
    def __init__(self, in_channels, mid_channels, out_channels, knl_list, strd_list,
                 pad_list, dilation_list, groups_list, bias_list=True):
        super(NLDeConv2d, self).__init__()
        self.out_channels_list = ModuleList([])

        for i in range(out_channels):
            kernel_size = self.getKernelSuparams(knl_list, i)
            stride = self.getKernelSuparams(strd_list, i)
            padding = self.getKernelSuparams(pad_list, i)
            dilation = self.getKernelSuparams(dilation_list, i)
            groups = self.getKernelSuparams(groups_list, i)
            bias = self.getKernelSuparams(bias_list, i)
            self.out_channels_list.append(
                self.build_deconvunit(in_channels=in_channels, mid_channels=mid_channels, kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias))

    def build_deconvunit(self, in_channels, mid_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True):
        module_list = ModuleList([])
        module_list.append(
            ConvTranspose2d(in_channels, mid_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,
                            bias=bias))
        module_list.append(LeakyReLU(0.1))
        module_list.append(Conv2d(mid_channels, 1, 1, 1, 0, dilation=1, groups=1, bias=True))
        ConvUnit = Sequential(*module_list)
        return ConvUnit

    def getKernelSuparams(self, params, index):
        """
        list or tuple and length must grater than 1.
        :param params:
        :param index:
        :return:
        """
        is_list = (isinstance(params, list) or isinstance(params, tuple)) and (len(params) > 1)
        # is_scalar = isinstance(params, int) or isinstance(params, bool) or (is_list and len(params)==1)
        if not is_list:
            param = params
        else:
            param = params[index]
        # else:
        #     raise NameError('params can only be `int` ,`list` or `tuple`. but %s were given. ' % type(params))
        return param

    def forward(self, input):
        results = []
        for unit in self.out_channels_list:
            results.append(unit(input))
        out = torch.cat(results, 1)
        return out


def Tconv3x3(in_planes, out_planes, mid_channels=16, stride=1, use_group=True, is_decomposed=False):
    "3x3 convolution with padding"

    repeat = out_planes // 2
    addition = out_planes % 2
    groups_list = []
    if stride == 1:
        addition_kpd = [3, 1, 1]
        if not is_decomposed:
            knl_list = [3, 3] * repeat
            strd_list = 1
            pad_list = [1, 2] * repeat
            dilation_list = [1, 2] * repeat
        if is_decomposed:
            knl_list = [(3, 1), (1, 3)] * repeat
            strd_list = 1
            pad_list = [(1, 0), (0, 2)] * repeat
            dilation_list = [1, 2] * repeat
    elif stride == 2:
        addition_kpd = [4, 2, 1]
        if not is_decomposed:
            # ksp: 421,220  1x4,4x1?
            knl_list = [4, 4] * repeat
            strd_list = 2
            pad_list = [1, 1] * repeat
            dilation_list = [1, 1] * repeat
        if is_decomposed:
            # ksp: 421,220  1x4,4x1?
            knl_list = [(1, 4), (4, 1)] * repeat
            strd_list = 2
            pad_list = [(0, 1), (1, 0)] * repeat
            dilation_list = [1, 1] * repeat
    else:
        raise AttributeError

    knl_list.extend([addition_kpd[0]] * addition)
    pad_list.extend([addition_kpd[1]] * addition)
    dilation_list.extend([addition_kpd[2]] * addition)
    assert len(dilation_list) == out_planes

    if use_group:
        assert (addition == 0), "you got out_channle %d. can not divided by 2" % out_planes
        for i in range(repeat):
            groups_list = groups_list + [1, 1] * (i + 1)
    else:
        groups_list = [1] * out_planes

    thickConv2d = NLConv2d(in_planes, mid_channels, out_planes,
                           knl_list=knl_list,
                           strd_list=strd_list,
                           pad_list=pad_list,
                           dilation_list=dilation_list,
                           groups_list=groups_list,
                           bias_list=False)
    return thickConv2d


def Tdeconv3x3(in_planes, out_planes, mid_channels=16, stride=1, use_group=True, is_decomposed=False):
    "3x3 convolution with padding"

    repeat = out_planes // 2
    addition = out_planes % 2
    groups_list = []
    if stride == 1:
        addition_kpd = [3, 1, 1]
        if not is_decomposed:
            knl_list = [3, 3] * repeat
            strd_list = 1
            pad_list = [1, 2] * repeat
            dilation_list = [1, 2] * repeat
        if is_decomposed:
            knl_list = [(3, 1), (1, 3)] * repeat
            strd_list = 1
            pad_list = [(1, 0), (0, 2)] * repeat
            dilation_list = [1, 2] * repeat
    elif stride == 2:
        addition_kpd = [4, 2, 1]
        if not is_decomposed:
            # ksp: 421,220  1x4,4x1?
            knl_list = [4, 4] * repeat
            strd_list = 2
            pad_list = [1, 1] * repeat
            dilation_list = [1, 1] * repeat
        if is_decomposed:
            # ksp: 421,220  1x4,4x1?
            knl_list = [(1, 4), (4, 1)] * repeat
            strd_list = 2
            pad_list = [(0, 1), (1, 0)] * repeat
            dilation_list = [1, 1] * repeat
    else:
        raise AttributeError

    knl_list.extend([addition_kpd[0]] * addition)
    pad_list.extend([addition_kpd[1]] * addition)
    dilation_list.extend([addition_kpd[2]] * addition)
    assert len(dilation_list) == out_planes

    if use_group:
        assert (addition == 0), "you got out_channle %d. can not divided by 2" % out_planes
        for i in range(repeat):
            groups_list = groups_list + [1, 1] * (i + 1)
    else:
        groups_list = [1] * out_planes

    thickConv2d = NLDeConv2d(in_planes, mid_channels, out_planes,
                             knl_list=knl_list,
                             strd_list=strd_list,
                             pad_list=pad_list,
                             dilation_list=dilation_list,
                             groups_list=groups_list,
                             bias_list=False)
    return thickConv2d


def InitTconv(in_planes, out_planes, mid_channels=16, use_group=True, is_decomposed=False):
    repeat = out_planes // 2
    addition = out_planes % 2
    groups_list = []
    addition_kpd = [4, 1, 0]

    if not is_decomposed:
        knl_list = [4, 4] * repeat
        strd_list = 1
        pad_list = [0, 0] * repeat
        dilation_list = [1, 1] * repeat
    else:
        knl_list = [(4, 1), (1, 4)] * repeat
        strd_list = 1
        pad_list = [(0, 0), (0, 0)] * repeat
        dilation_list = [1, 1] * repeat

    knl_list.extend([addition_kpd[0]] * addition)
    pad_list.extend([addition_kpd[1]] * addition)
    dilation_list.extend([addition_kpd[2]] * addition)
    assert len(dilation_list) == out_planes

    if use_group:
        assert (addition == 0), "you got out_channle %d. can not divided by 2" % out_planes
        for i in range(repeat):
            groups_list = groups_list + [1, 1] * (i + 1)
    else:
        groups_list = [1] * out_planes

    thickConv2d = NLDeConv2d(in_planes, mid_channels, out_planes,
                             knl_list=knl_list,
                             strd_list=strd_list,
                             pad_list=pad_list,
                             dilation_list=dilation_list,
                             groups_list=groups_list,
                             bias_list=False)
    return thickConv2d


class TconvLayer(Module):
    def __init__(self, c_in, c_out, mid_channels=16, stride=1, active_type="ReLU",
                 norm_type="batch", use_group=True,
                 is_decomposed=False, use_sn=False):
        super(TconvLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])

        if use_sn:
            self.module_list += [
                spectral_norm(
                    Tconv3x3(c_in, c_out, mid_channels, stride, use_group=use_group, is_decomposed=is_decomposed))]
        else:
            self.module_list += [
                Tconv3x3(c_in, c_out, mid_channels, stride, use_group=use_group, is_decomposed=is_decomposed)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class InitTconvlayer(Module):
    def __init__(self, c_in, c_out, mid_channels=16, active_type="ReLU",
                 norm_type="batch", use_group=True,
                 is_decomposed=False, use_sn=False):
        super(InitTconvlayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])

        if use_sn:
            self.module_list += [
                spectral_norm(
                    InitTconv(c_in, c_out, mid_channels, use_group=use_group, is_decomposed=is_decomposed))]
        else:
            self.module_list += [
                InitTconv(c_in, c_out, mid_channels, use_group=use_group, is_decomposed=is_decomposed)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class TdeconvLayer(Module):
    def __init__(self, c_in, c_out, mid_channels=16, stride=1, active_type="ReLU",
                 norm_type="batch", use_group=True,
                 is_decomposed=False, use_sn=False):
        super(TdeconvLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])

        if use_sn:
            self.module_list += [
                spectral_norm(
                    Tdeconv3x3(c_in, c_out, mid_channels, stride, use_group=use_group, is_decomposed=is_decomposed))]
        else:
            self.module_list += [
                Tdeconv3x3(c_in, c_out, mid_channels, stride, use_group=use_group, is_decomposed=is_decomposed)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


def getNormLayer(norm_type):
    norm_layer = BatchNorm2d
    if norm_type == 'batch':
        norm_layer = BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = InstanceNorm2d
    elif norm_type == 'switch':
        norm_layer = SwitchNorm
    elif norm_type == 'layer':
        norm_layer = LayerNorm
    elif norm_type == 'group':
        norm_layer = GroupNorm
    elif norm_type is None:
        norm_layer = None
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def getPoolLayer(pool_type):
    pool_layer = MaxPool2d(2, 2)
    if pool_type == 'Max':
        pool_layer = MaxPool2d(2, 2)
    elif pool_type == 'Avg':
        pool_layer = AvgPool2d(2, 2)
    elif pool_type is None:
        pool_layer = None
    else:
        print('pool layer [%s] is not found' % pool_layer)
    return pool_layer


def getActiveLayer(active_type):
    active_layer = ReLU
    if active_type == 'ReLU':
        active_layer = ReLU()
    elif active_type == 'LeakyReLU':
        active_layer = LeakyReLU(0.1)
    elif active_type == 'Tanh':
        active_layer = Tanh()
    elif active_type == 'Sigmoid':
        active_layer = Sigmoid()
    elif active_type is None:
        active_layer = None
    else:
        print('active layer [%s] is not found' % active_layer)
    return active_layer
