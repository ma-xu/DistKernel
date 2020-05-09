import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from torch.nn.parameter import Parameter
import torch
import time
from torch.nn.modules.utils import _single, _pair, _triple
# import torch.nn.functional as F
# from torch.nn import init
# from torch.autograd import Variable
# from collections import OrderedDict
import math
__all__ = ['combine12_resnet18', 'combine12_resnet34', 'combine12_resnet50', 'combine12_resnet101',
           'combine12_resnet152']


### FAIL!!! Due to the BN mean/avg
class AssConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,bias=False):
        super(AssConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.group = out_channels//32
        self.dilate = 2

        self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.second_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.dilate_conv = nn.Conv2d(in_channels, out_channels, math.ceil(kernel_size/2),
                                     stride, padding, dilation=2, bias=bias)
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=out_channels//32, bias=bias)
        self.ori_bn =  nn.BatchNorm2d(out_channels)
        self.dilate_bn = nn.BatchNorm2d(out_channels)
        self.group_bn = nn.BatchNorm2d(out_channels)
        self.second_bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('temp', torch.ones([1,in_channels,kernel_size,kernel_size]))

        self.fc = nn.Sequential(
            nn.Conv2d(4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 4, 1, bias=False)
        )
        # self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, input):
        ori_out = self.ori_bn(self.ori_conv(input)).unsqueeze(dim=1)
        second_out = self.second_bn(self.second_conv(input)).unsqueeze(dim=1)
        dilate_out = self.dilate_bn(self.dilate_conv(input)).unsqueeze(dim=1)
        group_out = self.group_bn(self.group_conv(input)).unsqueeze(dim=1)

        all_out = torch.cat([ori_out,second_out,dilate_out,group_out],dim=1) #N*4*C_out*W*H



        ori_temp_out = nn.functional.conv2d(self.temp,weight=self.ori_conv.weight.clone(),stride=_pair(self.stride))
        ori_temp_out = (ori_temp_out-self.ori_bn.running_mean.view_as(ori_temp_out))\
                       /(torch.sqrt(self.ori_bn.running_var + self.ori_bn.eps)).view_as(ori_temp_out)
        ori_temp_out = self.ori_bn.weight.clone().view_as(ori_temp_out)*ori_temp_out\
                       +self.ori_bn.bias.clone().view_as(ori_temp_out)

        second_temp_out = nn.functional.conv2d(self.temp, weight=self.second_conv.weight.clone(), stride=_pair(self.stride))
        second_temp_out = (second_temp_out - self.second_bn.running_mean.view_as(second_temp_out)) \
                       / (torch.sqrt(self.second_bn.running_var + self.second_bn.eps)).view_as(second_temp_out)
        second_temp_out = self.second_bn.weight.clone().view_as(second_temp_out) * second_temp_out \
                       + self.second_bn.bias.clone().view_as(second_temp_out)

        dilate_temp_out = nn.functional.conv2d(self.temp, weight=self.dilate_conv.weight.clone(),
                                               stride=_pair(self.stride),dilation=_pair(self.dilate))
        dilate_temp_out = (dilate_temp_out - self.dilate_bn.running_mean.view_as(dilate_temp_out)) \
                          / (torch.sqrt(self.dilate_bn.running_var + self.dilate_bn.eps)).view_as(dilate_temp_out)
        dilate_temp_out = self.dilate_bn.weight.clone().view_as(dilate_temp_out) * dilate_temp_out \
                          + self.dilate_bn.bias.clone().view_as(dilate_temp_out)

        group_temp_out = nn.functional.conv2d(self.temp, weight=self.group_conv.weight.clone(),
                                               groups=self.group)
        group_temp_out = (group_temp_out - self.group_bn.running_mean.view_as(group_temp_out)) \
                          / (torch.sqrt(self.group_bn.running_var + self.group_bn.eps)).view_as(group_temp_out)
        group_temp_out = self.group_bn.weight.clone().view_as(group_temp_out) * group_temp_out \
                          + self.group_bn.bias.clone().view_as(group_temp_out)

        all_temp_out = torch.cat([ori_temp_out, second_temp_out, dilate_temp_out, group_temp_out],
                                 dim=0).unsqueeze(dim=0).squeeze(dim=-1)

        alpha = self.fc(all_temp_out).unsqueeze(dim=-1) # 1*4*C_out*1*1
        alpha = alpha.softmax(dim=1) # 1*4*C_out*1*1
        out = alpha*all_out # N*4*C_out*W*H
        out = out.sum(dim=1,keepdim=False) # N*C_out*W*H
        return out











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


def combine12_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def combine12_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def combine12_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def combine12_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def combine12_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def demo():
    st = time.perf_counter()
    for i in range(1):
        net = combine12_resnet18(num_classes=1000)
        y = net(torch.randn(2, 3, 224,224))
        print(y.size())
    print("CPU time: {}".format(time.perf_counter() - st))

def demo2():
    st = time.perf_counter()
    for i in range(1):
        net = combine12_resnet50(num_classes=1000).cuda()
        y = net(torch.randn(2, 3, 224,224).cuda())
        print(y.size())
    print("CPU time: {}".format(time.perf_counter() - st))

# demo()
# demo2()

