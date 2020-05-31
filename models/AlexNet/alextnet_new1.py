import torch
import torch.nn as nn



__all__ = ['alexnet']



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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(AssConv, self).__init__()
        self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.ac_convbn = ACBlock(in_channels,out_channels,kernel_size,stride,padding)
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=64, bias=bias)

        self.ori_bn =  nn.BatchNorm2d(out_channels)
        self.new_bn = nn.BatchNorm2d(out_channels)
        self.group_bn = nn.BatchNorm2d(out_channels)


    def forward(self, input):
        return self.ori_bn(self.ori_conv(input))+self.new_bn(self.new_conv(input))\
               +self.ac_convbn(input)+self.group_bn(self.group_conv(input))



class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # AssConv(3,64, kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            AssConv(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            AssConv(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            AssConv(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            AssConv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    return model


def demo():
    net = alexnet(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
