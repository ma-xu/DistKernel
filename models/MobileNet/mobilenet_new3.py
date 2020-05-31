from torch import nn
import torch


__all__ = ['new3_mobilenet_v2']



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
    # in_planes, out_planes, kernel_size, stride, padding, groups = groups, bias = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,groups = 1, bias=False):
        super(AssConv, self).__init__()
        self.L = 3
        self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups, bias=bias)
        self.new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups, bias=bias)
        if kernel_size>1:
            self.ac_convbn = ACBlock(in_channels,out_channels,kernel_size,stride,padding,groups=groups)
            self.L=4
        if groups==1 and in_channels>=8:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=8, bias=bias)
        else:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        self.ori_bn =  nn.BatchNorm2d(out_channels)
        self.new_bn = nn.BatchNorm2d(out_channels)
        self.group_bn = nn.BatchNorm2d(out_channels)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // 4,3)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // 4,3), self.L),
            nn.Softmax(dim=1)
        )


    def forward(self, input):
        ori_out = self.ori_bn(self.ori_conv(input)).unsqueeze(dim=1)
        new_out = self.new_bn(self.new_conv(input)).unsqueeze(dim=1)
        group_out = self.group_bn(self.group_conv(input)).unsqueeze(dim=1)
        all_out = torch.cat([ori_out, new_out, group_out], dim=1)

        if hasattr(self, 'ac_convbn'):
            ac_out = self.ac_convbn(input).unsqueeze(dim=1)
            all_out = torch.cat([all_out, ac_out], dim=1)

        gap = input.mean(dim=-1).mean(dim=-1)
        weights = self.fc(gap).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        return out




        out =  self.ori_bn(self.ori_conv(input))+\
               self.new_bn(self.new_conv(input))+self.group_bn(self.group_conv(input))
        if hasattr(self, 'ac_convbn'):
            out =out+self.ac_convbn(input)
        return  out



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            # nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # nn.BatchNorm2d(out_planes),
            AssConv(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def new3_mobilenet_v2(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    return model



def demo():
    net = new3_mobilenet_v2(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
