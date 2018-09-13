# coding=utf-8
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d, ModuleList
import torch
from .shared.basic import *
from .shared.spectral_normalization import SpectralNorm


class Wnet_G(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=32, norm_type=None,
                 active_type="ReLU"):
        super(Wnet_G, self).__init__()
        self.depth = depth
        blocks = 6
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1

        self.input = normalBlock((input_nc, depth), (3, 1, 1), active_type=active_type,
                                 norm_type=norm_type,
                                 ksp_2=(3, 1, 1))
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []
        for i in range(num_hidden_blocks):
            if i >= num_hidden_blocks - 2:
                pool_type = "Avg"
            else:
                pool_type = "Max"
            self.add_module("downsample_block_" + str(i + 1),
                            downsampleBlock((self._depth(i), self._depth(i + 1)),
                                            (3, 1, 1),
                                            pool_type=pool_type,
                                            active_type=active_type, norm_type=norm_type))
            self.downsample += ["downsample_block_" + str(i + 1)]

        # bottle neck block
        self.bottleneck = normalBlock((max_depth, max_depth), (3, 1, 1), active_type=active_type,
                                      norm_type=norm_type)

        # up sample block 2-6, (4,3,2,1,0)
        self.upsample = []
        for i in range(num_hidden_blocks - 1, -1, -1):
            self.add_module("upsample_block_" + str(i + 1), upsampleBlock((self._depth(i + 2), self._depth(i)),
                                                                          (3, 1, 1),
                                                                          active_type=active_type,
                                                                          norm_type=norm_type))
            self.upsample += ["upsample_block_" + str(i + 1)]

        # output bloack
        self.output = normalBlock((depth, output_nc), (3, 1, 1), active_type="Sigmoid", ksp_2=(3, 1, 1))

    def _depth(self, i):
        return self.depth * (2 ** i)

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
            d_result.append(_input)

        _input = self.bottleneck(_input)

        # up sample block 2-6, (4,3,2,1,0)
        for name in self.upsample:
            prev_input = d_result.pop()
            _input = self._modules[name](prev_input, _input)
        # output
        out = self.output(_input)

        return out


class NLayer_D(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64, use_sigmoid=True, use_liner=True, norm_type="batch",
                 active_type="ReLU"):
        super(NLayer_D, self).__init__()
        self.norm = getNormLayer(norm_type)
        self.active = getActiveLayer(active_type)
        self.use_sigmoid = use_sigmoid
        self.use_liner = use_liner

        # 256 x 256
        self.layer1 = Sequential(SpectralNorm(Conv2d(input_nc + output_nc, depth, kernel_size=7, stride=1, padding=3)),
                                 LeakyReLU(0.1))
        # 128 x 128
        self.layer2 = Sequential(SpectralNorm(Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=1)),
                                 # self.norm(depth * 2, affine=True),
                                 LeakyReLU(0.1),
                                 MaxPool2d(2, 2))
        # 64 x 64
        self.layer3 = Sequential(SpectralNorm(Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=1)),
                                 # self.norm(depth * 4, affine=True),
                                 LeakyReLU(0.1),
                                 MaxPool2d(2, 2))
        # 32 x 32
        self.layer4 = Sequential(SpectralNorm(Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=1)),
                                 # self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.1),
                                 AvgPool2d(2, 2))
        # 16 x 16
        self.layer5 = Sequential(SpectralNorm(Conv2d(depth * 8, depth * 16, kernel_size=3, stride=1, padding=1)),
                                 # self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.1),
                                 AvgPool2d(2, 2))
        # # 8 x 8
        # self.layer6 = Sequential(SpectralNorm(Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=1)),
        #                          # self.norm(depth * 8, affine=True),
        #                          LeakyReLU(0.2),
        #                          MaxPool2d(2,2))
        # # 4 x 4
        # self.layer7 = Sequential(SpectralNorm(Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1)),
        #                          # self.norm(depth * 8, affine=True),
        #                          LeakyReLU(0.2))
        # 8 x 8
        self.layer6 = Sequential(Conv2d(depth * 16, output_nc, kernel_size=8, stride=1, padding=1))
        # 16 x 16 ,1
        self.liner = Linear(256, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.use_liner:
            out = out.view(out.size(0), -1)
            out = self.liner(out)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out


class Branch_Wnet_G(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=32, norm_type=None,
                 active_type="ReLU", branches=3):
        super(Branch_Wnet_G, self).__init__()
        self.depth = depth
        self.branches = branches
        blocks = 6
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1

        self.input = normalBlock((input_nc, depth), (7, 1, 3), active_type=active_type,
                                 norm_type=norm_type,
                                 ksp_2=(5, 1, 2))
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []
        for i in range(num_hidden_blocks):
            if i >= num_hidden_blocks - 2:
                pool_type = "Avg"
            else:
                pool_type = "Max"
            self.add_module("downsample_block_" + str(i + 1),
                            downsampleBlock((self._depth(i), self._depth(i + 1)),
                                            (3, 1, 1),
                                            pool_type=pool_type,
                                            active_type=active_type))
            self.downsample += ["downsample_block_" + str(i + 1)]

        # bottle neck block
        self.bottleneck = normalBlock((max_depth, max_depth), (3, 1, 1), active_type=active_type,
                                      norm_type=norm_type)

        # up sample block 2-6, (4,3,2,1,0)
        self.upsample = []
        for i in range(num_hidden_blocks - 1, -1, -1):
            if i >= 2:
                groups = 1
            else:
                groups = branches
            self.add_module("upsample_block_" + str(i + 1), upsampleBlock((self._depth(i + 2), self._depth(i)),
                                                                          (3, 1, 1),
                                                                          active_type=active_type,
                                                                          norm_type=norm_type, groups=groups))
            self.upsample += ["upsample_block_" + str(i + 1)]

        # output bloack
        self.output = normalBlock((depth, branches), (7, 1, 3), active_type="Tanh", ksp_2=(5, 1, 2), groups=branches)

    def _depth(self, i):
        return self.depth * (2 ** i)

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
            d_result.append(_input)

        _input = self.bottleneck(_input)

        # up sample block 2-6, (4,3,2,1,0)
        for name in self.upsample:
            prev_input = d_result.pop()
            _input = self._modules[name](prev_input, _input)
        # output
        out = self.output(_input)

        return out


class Attn_Wnet_G(Module):
    def __init__(self, depth=32, norm_type=None, active_type="ReLU", use_sn=True):
        super(Attn_Wnet_G, self).__init__()
        self.depth = depth
        input_nc = 1
        output_nc = 1
        blocks = 6
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1
        self.input = normalBlock((input_nc, depth), (3, 1, 1), active_type=active_type,
                                 norm_type=norm_type, use_sn=use_sn)
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []

        for i in range(num_hidden_blocks):
            if i <=3:
                pool_type = "Max"
            else:
                pool_type = "Avg"
            self.add_module("downsample_block_" + str(i + 1),
                            downsampleBlock((self._depth(i), self._depth(i + 1)),
                                            (3, 1, 1),
                                            pool_type=pool_type,
                                            active_type=active_type,
                                            norm_type=norm_type,
                                            use_sn=use_sn))
            self.downsample += ["downsample_block_" + str(i + 1)]

        # bottle neck block
        self.bottleneck = normalBlock((max_depth, max_depth), (3, 1, 1),
                                      active_type=active_type,
                                      norm_type=norm_type,
                                      use_sn=use_sn)

        # up sample block 2-6, (4,3,2,1,0)
        self.upsample = []
        self.attention = []
        for i in range(num_hidden_blocks - 1, -1, -1):

            self.add_module("upsample_block_" + str(i + 1), upsampleBlock((self._depth(i + 2), self._depth(i)),
                                                                          (3, 1, 1),
                                                                          active_type=active_type,
                                                                          norm_type=norm_type, use_sn=use_sn))
            # if i ==4 :
            #     self.add_module("self_attention_" + str(i + 1), Self_Attn(self._depth(i), 'relu', False))
            #     self.attention += ["self_attention_" + str(i + 1)]
            # else:
            #     self.attention += ["None"]

            self.upsample += ["upsample_block_" + str(i + 1)]

        # output bloack
        self.output = normalBlock((depth, output_nc), (3, 1, 1), active_type="Tanh", use_sn=False)

    def _depth(self, i):
        return self.depth * (2 ** i)

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
            d_result.append(_input)

        _input = self.bottleneck(_input)

        # up sample block 2-6, (4,3,2,1,0)
        for index, name in enumerate(self.upsample):
            prev_input = d_result.pop()
            _input = self._modules[name](prev_input, _input)
            # if index == 0 :
            #     _input =self._modules[self.attention[index]]( _input)
                # output
        out = self.output(_input)

        return out


class Attn_Discriminator(Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, depth=32):
        super(Attn_Discriminator, self).__init__()
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        # layer5 = []
        last = []

        layer1.append(SpectralNorm(Conv2d(2, depth, 4, 2, 1)))
        layer1.append(LeakyReLU(0.1))

        layer2.append(SpectralNorm(Conv2d(depth, depth * 2, 4, 2, 1)))
        layer2.append(LeakyReLU(0.1))

        layer3.append(SpectralNorm(Conv2d(depth * 2, depth * 4, 4, 2, 1)))
        layer3.append(LeakyReLU(0.1))

        layer4.append(SpectralNorm(Conv2d(depth * 4, depth * 8, 4, 2, 1)))
        layer4.append(LeakyReLU(0.1))

        # layer5.append(SpectralNorm(Conv2d(depth * 8, depth * 16, 4, 2, 1)))
        # layer5.append(LeakyReLU(0.1))
        # depth = depth * 2

        self.l1 = Sequential(*layer1)
        self.l2 = Sequential(*layer2)
        self.l3 = Sequential(*layer3)
        self.l4 = Sequential(*layer4)
        # self.l5 = Sequential(*layer5)

        last.append(Conv2d(depth * 8, 1, 4))
        self.last = Sequential(*last)

        self.attn1 = Self_Attn(depth * 4, 'relu')
        self.attn2 = Self_Attn(depth * 8, 'relu')
        # self.attn3 = Self_Attn(depth * 16, 'relu')

    def forward(self, x, input):
        out = torch.cat((x, input),1)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        # out = self.l5(out)
        # out, p3 = self.attn3(out)
        out = self.last(out)

        return out.squeeze()


class Self_Attn(Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, show_attention=True):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.show_attention = show_attention
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        if self.show_attention:
            return out, attention
        else:
            return out


class Branch_NLayer_D(Module):
    def __init__(self, input_nc=1, output_nc=3, depth=64, use_sigmoid=True, norm_type="batch",
                 active_type="ReLU", groups=1):
        super(Branch_NLayer_D, self).__init__()
        self.use_sigmoid = use_sigmoid

        # 256 x 256
        self.layer1 = convLayer(input_nc + input_nc, depth, 8, 2, 3, active_type=active_type, norm_type=None,
                                groups=groups)

        # 128 x 128
        self.layer2 = convLayer(depth, depth * 2, 4, 2, 1, active_type=active_type, norm_type=norm_type,
                                groups=groups)

        # 64 x 64
        self.layer3 = convLayer(depth * 2, depth * 4, 4, 2, 1, active_type=active_type, norm_type=norm_type,
                                groups=groups)

        # 32 x 32
        self.layer4 = convLayer(depth * 4, depth * 8, 4, 2, 1, active_type=active_type, norm_type=norm_type,
                                groups=groups)

        # 16 x 16
        self.layer5 = Conv2d(depth * 8, output_nc, kernel_size=5, stride=1, padding=2)

        # 16 x 16 ,1
        self.liner = Linear(256, 1)

        self.sigmoid = Sigmoid()

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        if self.use_sigmoid:
            out = self.sigmoid(out)

        return out
