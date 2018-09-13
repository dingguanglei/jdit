from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d
import torch
from .shared.basic import *
import numpy as np


class NThickLayer_D(Module):
    def __init__(self, input_nc=1, depth=64):
        super(NThickLayer_D, self).__init__()

        unit_cn = 12
        # 32 x 32 x 12  k-2p =s   7 1 3, 5 1 2, 3 1 1, 1,1,0
        self.layer1 = Sequential(ThickVarietyConv2d(input_nc, depth * 1, unit_cn,
                                                    knl_list=[1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5],  # 1,3,5,3,7,11
                                                    strd_list=1,
                                                    pad_list=[0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 2, 4],
                                                    dilation_list=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                                                    groups_list=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                 LeakyReLU(0.1))
        # 32 x 32 x 24
        self.layer2 = Sequential(ThickVarietyConv2d(unit_cn, depth * 1, unit_cn * 1,
                                                    knl_list=[1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5],  # 1,3,5,3,7,11
                                                    strd_list=1,
                                                    pad_list=[0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 2, 4],
                                                    dilation_list=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                                                    groups_list=[1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4]),
                                 LeakyReLU(0.1))
        # 32 x 32 x 48
        self.layer3 = Sequential(ThickVarietyConv2d(unit_cn * 1, depth * 1, unit_cn * 1,
                                                    knl_list=[1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5],  # 1,3,5,3,7,11
                                                    strd_list=1,
                                                    pad_list=[0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 2, 4],
                                                    dilation_list=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                                                    groups_list=[1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4]),
                                 LeakyReLU(0.1))
        # 32 x 32 x 96
        self.layer4 = Sequential(ThickVarietyConv2d(unit_cn * 1, depth * 1, unit_cn * 1,
                                                    knl_list=[1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5],  # 1,3,5,3,7,11
                                                    strd_list=1,
                                                    pad_list=[0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 2, 4],
                                                    dilation_list=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                                                    groups_list=[1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4])
                                 )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class NLayer_D(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64, use_sigmoid=True, use_liner=True, norm_type="batch",
                 active_type="ReLU"):
        super(NLayer_D, self).__init__()
        self.norm = getNormLayer(norm_type)
        self.active = getActiveLayer(active_type)
        self.use_sigmoid = use_sigmoid
        self.use_liner = use_liner

        # 32 x 32
        self.layer1 = Sequential(Conv2d(input_nc, depth * 1, kernel_size=3, stride=1, padding=1),
                                 # self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.1),
                                 MaxPool2d(2, 2))
        # 16 x 16
        self.layer2 = Sequential(Conv2d(depth * 1, depth * 2, kernel_size=3, stride=1, padding=1),
                                 # self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.1),
                                 MaxPool2d(2, 2))
        # 8 x 8
        self.layer3 = Sequential(Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=1),
                                 # self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.1),
                                 MaxPool2d(2, 2))
        # 4 x 4
        # self.layer4 = Sequential(Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=1),
        # self.norm(depth * 8, affine=True),
        # LeakyReLU(0.1),
        # AvgPool2d(2, 2))
        # 8 x 8
        self.layer4 = Sequential(Conv2d(depth * 4, output_nc, kernel_size=4, stride=1, padding=0))
        # 16 x 16 ,1
        self.liner = Linear(256, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # out = torch.cat((x, g), 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.layer4(out)

        if self.use_liner:
            out = out.view(out.size(0), -1)
            out = self.liner(out)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out


class NThickClassLayer_D(Module):
    def __init__(self, input_nc=1, depth=64):
        super(NThickClassLayer_D, self).__init__()

        unit_cn = 12
        knl_list = [1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5]
        strd_list = 1
        pad_list = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        dilation_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        groups_list = [1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4]

        # 32 x 32 x 12  k-2p =s   7 1 3, 5 1 2, 3 1 1, 1,1,0
        self.layer1 = Sequential(ThickVarietyConv2d(input_nc, depth * 1, unit_cn,
                                                    knl_list=np.add(knl_list, 1).tolist(),  # 1,3,5,3,7,11
                                                    strd_list=strd_list + 1,
                                                    pad_list=pad_list,
                                                    dilation_list=dilation_list,
                                                    groups_list=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                 LeakyReLU(0.1))
        # 16 x 16 x 24
        self.layer2 = Sequential(ThickVarietyConv2d(unit_cn, depth * 1, unit_cn * 1,
                                                    knl_list=np.add(knl_list, 1).tolist(),  # 1,3,5,3,7,11
                                                    strd_list=strd_list + 1,
                                                    pad_list=pad_list,
                                                    dilation_list=dilation_list,
                                                    groups_list=groups_list),
                                 LeakyReLU(0.1))
        # 8 x 8 x 48
        self.layer3 = Sequential(ThickVarietyConv2d(unit_cn * 1, depth * 1, unit_cn * 1,
                                                    knl_list=np.add(knl_list, 1).tolist(),  # 1,3,5,3,7,11
                                                    strd_list=strd_list + 1,
                                                    pad_list=pad_list,
                                                    dilation_list=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    groups_list=groups_list),
                                 LeakyReLU(0.1))
        # 4 x 4 x 96
        self.layer4 = Sequential(Conv2d(12, 10, 4, 1, 0), Sigmoid())
        # 1 x 1 x 10

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 10)
        return out


class NNormalClassLayer_D(Module):
    def __init__(self, input_nc=1, depth=64):
        super(NNormalClassLayer_D, self).__init__()

        # 32 x 32
        self.layer1 = Sequential(Conv2d(input_nc, depth * 1, kernel_size=4, stride=2, padding=1),
                                 LeakyReLU(0.1))
        # 16 x 16
        self.layer2 = Sequential(Conv2d(depth * 1, depth * 2, kernel_size=4, stride=2, padding=1),
                                 LeakyReLU(0.1))
        # 8 x 8
        self.layer3 = Sequential(Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1),
                                 LeakyReLU(0.1))
        # 4 x 4
        self.layer4 = Sequential(Conv2d(depth * 4, 10, kernel_size=4, stride=1, padding=0))
        # 1 x 1 ,10
        self.sigmoid = Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 10)
        return out


class TWnet_G(Module):
    def __init__(self, input_nc=16, output_nc=1, depth=32, norm_type=None,
                 active_type="ReLU"):
        super(TWnet_G, self).__init__()
        self.depth = depth
        # input_nc = depth * 2
        # 4 * 4 * 16
        self.layer1 = convLayer(input_nc, depth * 8, 3, 1, 1, active_type=active_type,
                                norm_type=norm_type, groups=1, use_sn=True)
        # 4 * 4 * 32*8
        self.layer2 = deconvLayer(depth * 8, depth * 4, 4, 2, 1, active_type=active_type,
                                  norm_type=norm_type, groups=1, use_sn=True)
        # 8 * 8 * 32*4
        self.layer3 = deconvLayer(depth * 4, depth * 2, 4, 2, 1, active_type=active_type,
                                  norm_type=norm_type, groups=1, use_sn=True)
        # 16 * 16 * 32*2
        self.layer4 = deconvLayer(depth * 2, depth * 1, 4, 2, 1, active_type=active_type,
                                  norm_type=norm_type, groups=1, use_sn=True)
        # 32 * 32 * 32*1
        self.output = deconvLayer(depth * 1, output_nc, 3, 1, 1, active_type="Tanh",
                                  norm_type=norm_type, groups=1, use_sn=False)
        # 32 * 32 * 1

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.output(out)
        return out
