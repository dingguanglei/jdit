import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
import math
from .shared.basic import ThickVarietyConv2d


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lrelu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lrelu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def Tconv3x3(in_planes, out_planes, mid_channels=32, stride=1, use_group=True, type="normal"):
    "3x3 convolution with padding"

    """
    if stride == 1:
        knl_list = [3, 5, 3, 5] * (out_planes // 4)
        strd_list = 1
        pad_list = [1, 2, 2, 4] * (out_planes // 4)
        dilation_list = [1, 1, 2, 2] * (out_planes // 4)
        groups_list = []
    else:
        knl_list = [4, 6, 4, 6] * (out_planes // 4)
        strd_list = 2
        pad_list = [1, 2, 1, 2] * (out_planes // 4)
        dilation_list = [1, 1, 1, 1] * (out_planes // 4)
        groups_list = []
    """
    """
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
    """
    repeat = out_planes // 2
    groups_list = []
    if stride == 1 and type == "normal":
        knl_list = [3, 3] * repeat
        strd_list = 1
        pad_list = [1, 2] * repeat
        dilation_list = [1, 2] * repeat

    elif stride == 1 and type == "decomposition":
        knl_list = [(3, 1), (1, 3)] * repeat
        strd_list = 1
        pad_list = [(1, 0), (0, 2)] * repeat
        dilation_list = [1, 2] * repeat

    elif stride == 2 and type == "normal":
        # ksp: 421,220  1x4,4x1?
        knl_list = [4, 4] * repeat
        strd_list = 2
        pad_list = [1, 1] * repeat
        dilation_list = [1, 1] * repeat

    elif stride == 2 and type == "decomposition":
        # ksp: 421,220  1x4,4x1?
        knl_list = [(1, 4), (4, 1)] * repeat
        strd_list = 2
        pad_list = [(0, 1), (1, 0)] * repeat
        dilation_list = [1, 1] * repeat

    else:
        raise AttributeError

    if use_group:
        for i in range(repeat):
            groups_list = groups_list + [1, 1] * (i + 1)
    else:
        groups_list = [1, 1] * repeat

    thickConv2d = ThickVarietyConv2d(in_planes, mid_channels, out_planes,
                                     knl_list=knl_list,
                                     strd_list=strd_list,
                                     pad_list=pad_list,
                                     dilation_list=dilation_list,
                                     groups_list=groups_list,
                                     bias_list=False)
    return thickConv2d


class TBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, mid_channels=32, stride=1, downsample=None, type="normal"):
        super(TBasicBlock, self).__init__()
        self.conv1 = Tconv3x3(in_planes, out_planes, mid_channels, stride, type=type)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = Tconv3x3(out_planes, out_planes, mid_channels, type=type)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride
        self.drop2d = nn.Dropout2d(p=0.2)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop2d(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2d(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, depth=16, mid_channels=32):
        self.mid_channels = mid_channels
        super(TResNet, self).__init__()

        self.conv1 = Tconv3x3(1, depth, mid_channels, stride=1, use_group=False, type="normal")
        self.bn1 = nn.BatchNorm2d(depth)
        self.relu = nn.LeakyReLU(0.1)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, depth * 1, depth * 1, layers[0], stride=1, type="normal")
        self.layer2 = self._make_layer(block, depth * 1, depth * 2, layers[1], stride=2, type="decomposition")
        self.layer3 = self._make_layer(block, depth * 2, depth * 4, layers[2], stride=2, type="decomposition")
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(depth * 4, num_classes)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, type="normal"):
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        layers.append(block(in_planes, out_planes, self.mid_channels, stride, downsample=downsample, type=type))
        # self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(out_planes, out_planes, self.mid_channels, stride=1, type=type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


ResNet18 = resnet18(False)

TResNet18 = TResNet(TBasicBlock, [2, 2, 2])


def Tresnet18(depth, mid_channels):
    TResNet18 = TResNet(TBasicBlock, [2, 2, 2], 10, depth, mid_channels)
    return TResNet18
