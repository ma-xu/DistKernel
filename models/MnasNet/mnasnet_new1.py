import torch
import torch.nn as nn

__all__ = [ 'new1_mnasnet0_5', 'new1_mnasnet0_75', 'new1_mnasnet1_0', 'new1_mnasnet1_3']


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', bn_momentum=0.1):
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
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels,momentum=bn_momentum)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels,momentum=bn_momentum)



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
    # in_planes, out_planes, kernel_size, stride, padding, groups = groups, bias = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,groups = 1, bias=False,bn_momentum=0.1):
        super(AssConv, self).__init__()
        self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups, bias=bias)
        self.new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups, bias=bias)
        if kernel_size>1:
            self.ac_convbn = ACBlock(in_channels,out_channels,kernel_size,stride,padding,groups=groups)
        if groups==1 and in_channels>=8:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=8, bias=bias)
        else:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        self.ori_bn =  nn.BatchNorm2d(out_channels,momentum=bn_momentum)
        self.new_bn = nn.BatchNorm2d(out_channels,momentum=bn_momentum)
        self.group_bn = nn.BatchNorm2d(out_channels,momentum=bn_momentum)


    def forward(self, input):
        out =  self.ori_bn(self.ori_conv(input))+\
               self.new_bn(self.new_conv(input))+self.group_bn(self.group_conv(input))
        if hasattr(self, 'ac_convbn'):
            out =out+self.ac_convbn(input)
        return  out





class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor,
                 bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            # nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            # nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            AssConv(in_ch,mid_ch,1,bias=False,bn_momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            # nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
            #           stride=stride, groups=mid_ch, bias=False),
            # nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            AssConv(mid_ch, mid_ch, kernel_size,padding=kernel_size // 2,
                    stride=stride,groups=mid_ch, bias=False, bn_momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            # nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            # nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            AssConv(mid_ch,out_ch,1,bias=False,bn_momentum=bn_momentum)
        )

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats,
           bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor,
                              bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor,
                              bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _scale_depths(depths, alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    # """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf.
    # >>> model = MNASNet(1000, 1.0)
    # >>> x = torch.rand(1, 3, 224, 224)
    # >>> y = model(x)
    # >>> y.dim()
    # 1
    # >>> y.nelement()
    # 1000
    # """

    def __init__(self, alpha, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        depths = _scale_depths([24, 40, 80, 96, 192, 320], alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[5], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)


def new1_mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    return model


def new1_mnasnet0_75(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    return model


def new1_mnasnet1_0(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, **kwargs)
    return model


def new1_mnasnet1_3(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, **kwargs)
    return model

def demo():
    net = new1_mnasnet1_0(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
