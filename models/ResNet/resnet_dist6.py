"""
Kernel_distribution + Kernel_perturbation
    Kernel_distribution : kaiming_normal_ init for each channel, channel are repeated.
                          this may leads the var(w_c) not queals to 2/n_l, since smaples to little, e.g. 9.
    Kernel_perturbation : 0 init
"""

import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import time
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
# from torch.nn import init
# from torch.autograd import Variable
# from collections import OrderedDict
# from torch.distributions.normal import Normal
import math
__all__ = ['dist6_resnet18', 'dist6_resnet34', 'dist6_resnet50', 'dist6_resnet101',
           'dist6_resnet152']



class DPConv(nn.Module):
    def __init__(self, in_planes, out_planes, k=3, stride=1):
        super(DPConv, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.k = k
        self.stride = stride
        self.perturbation = nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.register_buffer('mask', self._get_mask())
        self.distribution_zoom = Parameter(torch.ones(1,1,self.out_planes,self.in_planes)) # for mask size [-1,0,1]*zoom
        self.distribution_std = Parameter(0.001*torch.ones(out_planes,in_planes)) # for normal distribution variance.
        self.distribution_bias = Parameter(torch.zeros(out_planes, in_planes, 1, 1))
        self.distribution_scale = Parameter(torch.zeros(self.out_planes,self.in_planes,1,1))



    def forward(self, input):
        # print(self.mask)
        param = self._init_distribution()

        distribution_out = self._distribution_conv(input,param)

        return self.bn1(distribution_out)+self.bn2(self.perturbation(input))

    def _get_mask(self):
        # assume square and odd number, because lazy.
        y = torch.arange(0, self.k) - self.k // 2 + 0.0
        y1 = y.reshape(self.k, 1).expand((self.k, self.k))
        y2 = y.reshape(1, self.k).expand((self.k, self.k))
        mask = torch.sqrt(y1.pow(2) + y2.pow(2))
        return mask.unsqueeze(dim=-1).unsqueeze(dim=-1)


    def _init_distribution(self):
        # self.distribution_std
        std = F.relu(self.distribution_std)+1e-5
        y = -(1.0/(std*math.sqrt(2*math.pi)))\
            *torch.exp(-((self.mask*self.distribution_zoom)**2)/(2*(std**2)))
        y = y.permute(2, 3, 0, 1)
        y = self.distribution_scale * y + self.distribution_bias
        return y


    def _distribution_conv(self,input, weight,bias=None,
                 padding=1, dilation=1, groups=1):
        stride = _pair(self.stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)


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
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.dpConv1 = DPConv(inplanes, planes,stride=stride)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.dpConv2 = DPConv(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        out = self.dpConv1(x)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        out = self.dpConv2(out)

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
        # self.conv2 = conv3x3(planes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.dpConv = DPConv(planes, planes,stride=stride)
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

        # out = self.conv2(out)
        # out = self.bn2(out)
        out = self.dpConv(out)
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

        # # For DPConv init.
        # for m in self.modules():
        #     if isinstance(m,DPConv):
        #         # m.distribution.weight = Parameter(self._dstribution_norm_(m.distribution.weight))
        #         nn.init.constant_(m.perturbation.weight, 0)

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

    def _dstribution_norm_(self, tensor):
        _,fanout = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(fanout)
        # normal = Normal(loc=0.,scale=std)
        with torch.no_grad():
            return (tensor[0,0,:,:].normal_(0, std)).expand_as(tensor)


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


def dist6_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def dist6_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def dist6_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def dist6_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def dist6_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def demo():
    st = time.perf_counter()
    for i in range(1):
        net = dist6_resnet50(num_classes=1000)
        y = net(torch.randn(2, 3, 224,224))
        print(y.size())
        # for name, param in net.state_dict().items():
        #     # print(name)
        #     if "distribution" in name:
        #         print("{}: {}".format(name,param))
        #     if "perturbation" in name:
        #         print("{}: {}".format(name,param))

    print("CPU time: {}".format(time.perf_counter() - st))



def demo2():
    st = time.perf_counter()
    for i in range(50):
        net = dist6_resnet50(num_classes=1000).cuda()
        y = net(torch.randn(2, 3, 224,224).cuda())
        print(y.size())
    print("CPU time: {}".format(time.perf_counter() - st))

demo()
# demo2()

