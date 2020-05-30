import torch
import sys
sys.path.append("../")
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import models as model
import cv2
from models.ResNet.resnet_new1 import *
from models.ResNet.resnet_new3 import *

# img_path = "/Users/melody/Downloads/cat.jpeg"
img_path = "/Users/melody/Downloads/bird2.jpeg"
path = "/Users/melody/Desktop/Experiments/ParaDise/new1_resnet18_FP32_model_best.pth.tar"
path = "/Users/melody/Desktop/Experiments/ParaDise/new3_resnet18_FP32_model_best.pth.tar"
check_point = torch.load(path,map_location=torch.device('cpu') )
check_point = check_point['state_dict']
new_check = OrderedDict()
for k, v in check_point.items():
    name = k[7:]  # remove `module.`
    new_check[name] = v


def preprocess_image(img_path):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    return preprocessed_img

x = preprocess_image(img_path)
# net = new1_resnet18()
net = new3_resnet18()
net.load_state_dict(new_check)
# print(net)

def get_features_hook(self, input, output):
    # input is a tuple.
    out_ori = self.ori_bn(self.ori_conv(input[0])).mean(dim=-1).mean(dim=-1)
    out_new = self.new_bn(self.new_conv(input[0])).mean(dim=-1).mean(dim=-1)
    out_ac = self.ac_convbn(input[0]).mean(dim=-1).mean(dim=-1)
    out_group =self.group_bn(self.group_conv(input[0])).mean(dim=-1).mean(dim=-1)
    min_value = min(out_ori.min().item(), out_new.min().item(), out_ac.min().item(), out_group.min().item())
    max_value = max(out_ori.max().item(), out_new.max().item(), out_ac.max().item(), out_group.max().item())

    ori_hist = torch.histc(out_ori, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
    new_hist = torch.histc(out_new, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
    group_hist = torch.histc(out_group, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
    ac_hist = torch.histc(out_ac, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)

    print(ori_hist.int())
    print(new_hist.int())
    print(group_hist.int())
    print(ac_hist.int())
    print(min_value)
    print(max_value)
layer = net.layer4[0].conv2
handle=layer.register_forward_hook(get_features_hook)

y = net(x)
print(x.shape)

