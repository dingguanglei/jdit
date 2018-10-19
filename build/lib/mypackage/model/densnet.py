import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared.basic import NLConv2d
import math


def Tconv3x3(in_planes, out_planes, mid_channels=32, stride=1, use_group=True):
    "3x3 convolution with padding"
    repeat = out_planes // 2
    if stride == 1:
        knl_list = [3, 3] * repeat
        strd_list = 1
        pad_list = [1, 2] * repeat
        dilation_list = [1, 2] * repeat
        groups_list = []
    else:
        # ksp: 421,220  1x4,4x1?
        knl_list = [4, 4] * repeat
        strd_list = 2
        pad_list = [1, 1] * repeat
        dilation_list = [1, 1] * repeat
        groups_list = []

    if use_group:
        for i in range(repeat):
            groups_list = groups_list + [1, 1] * (i + 1)
    else:
        groups_list = [1, 1] * repeat

    thickConv2d = NLConv2d(in_planes, mid_channels, out_planes,
                           knl_list=knl_list,
                           strd_list=strd_list,
                           pad_list=pad_list,
                           dilation_list=dilation_list,
                           groups_list=groups_list,
                           bias_list=False)
    return thickConv2d


class T_SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, mid_channels=32):
        super(T_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = Tconv3x3(nChannels, growthRate, mid_channels, stride=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class T_Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, mid_channels=32):
        super(T_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        # self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
        #                        bias=False)

        self.conv1 = Tconv3x3(nChannels, nOutChannels, mid_channels, stride=2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # out = F.avg_pool2d(out, 2)
        return out


class TDenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck=False, mid_channels=16):
        super(TDenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = Tconv3x3(1, nChannels, mid_channels, stride=1, use_group=False)
        # nn.Conv2d(3, nChannels, kernel_size=3, padding=1,bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)

        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(T_SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        out = self.fc(out)
        return out


# ——————————————————————————————————————————————

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)

        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        out = self.fc(out)
        return out


# denseNet = DenseNet(growthRate=12, depth=100, reduction=0.5,
#                             bottleneck=False, nClasses=10)

# TdenseNet = TDenseNet(growthRate=12, depth=100, reduction=0.5,
#                             bottleneck=False, nClasses=10, mid_channels=16)
def denseNet(growthRate=12, depth=100):
    denseNet = DenseNet(growthRate=growthRate, depth=depth, reduction=0.5,
                        bottleneck=False, nClasses=10)
    return denseNet


def TdenseNet(growthRate=12, depth=100, mid_channels=16):
    TdenseNet = TDenseNet(growthRate=growthRate, depth=depth, reduction=0.5,
                          bottleneck=False, nClasses=10, mid_channels=mid_channels)
    return TdenseNet
