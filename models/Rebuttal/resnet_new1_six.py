import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from torch.nn.parameter import Parameter
import torch
# import time
# import torch.nn.functional as F
# from torch.nn import init
# from torch.autograd import Variable
# from collections import OrderedDict
import math
__all__ = ['new1_six_resnet18', 'new1_six_resnet34', 'new1_six_resnet50', 'new1_six_resnet101',
           'new1_six_resnet152']

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
        if center_offset_from_origin_border >= 0:
            self.ver_conv_crop_layer = nn.Identity()
            ver_conv_padding = ver_pad_or_crop
            self.hor_conv_crop_layer = nn.Identity()
            hor_conv_padding = hor_pad_or_crop
        else:
            self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
            ver_conv_padding = (0, 0)
            self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
            hor_conv_padding = (0, 0)
        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                  stride=stride,
                                  padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                  stride=stride,
                                  padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels)



    def forward(self, input):
        # print(square_outputs.size())
        # return square_outputs
        vertical_outputs = self.ver_conv_crop_layer(input)
        vertical_outputs = self.ver_conv(vertical_outputs)
        vertical_outputs = self.ver_bn(vertical_outputs)
        # print(vertical_outputs.size())
        horizontal_outputs = self.hor_conv_crop_layer(input)
        horizontal_outputs = self.hor_conv(horizontal_outputs)
        horizontal_outputs = self.hor_bn(horizontal_outputs)
        # print(horizontal_outputs.size())
        return vertical_outputs + horizontal_outputs


class AssConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(AssConv, self).__init__()
        self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.ac_convbn = ACBlock(in_channels,out_channels,kernel_size,stride,padding)
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=64, bias=bias)

        self.ori_bn =  nn.BatchNorm2d(out_channels)
        self.new_bn = nn.BatchNorm2d(out_channels)
        self.group_bn = nn.BatchNorm2d(out_channels)


        # double it
        self.ac_convbn_2 = ACBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.group_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=64, bias=bias)
        self.group_bn_2 = nn.BatchNorm2d(out_channels)


    def forward(self, input):
        return self.ori_bn(self.ori_conv(input))+self.new_bn(self.new_conv(input))\
               +self.ac_convbn(input)+self.group_bn(self.group_conv(input))\
               +self.ac_convbn_2(input)+self.group_bn_2(self.group_conv_2(input))

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = AssConv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = AssConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AssConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def new1_six_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def new1_six_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def new1_six_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def new1_six_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def new1_six_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def demo():
    # st = time.perf_counter()
    for i in range(1):
        net = new1_six_resnet18(num_classes=1000)
        y = net(torch.randn(2, 3, 224,224))
        print(y.size())
    # print("CPU time: {}".format(time.perf_counter() - st))

def demo2():
    # st = time.perf_counter()
    for i in range(1):
        net = new1_six_resnet50(num_classes=1000).cuda()
        y = net(torch.randn(2, 3, 224,224).cuda())
        print(y.size())
    # print("CPU time: {}".format(time.perf_counter() - st))

# demo()
# demo2()

