
import torch
import sys
sys.path.append('../')
import models as models
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-p', '--path', default='rnew1_resnet50', type=str)
args = parser.parse_args()


checkpoint_path ='/home/g1007540910/DistKernel/checkpoints/imagenet/'+args.path+'/'
load_path = checkpoint_path + 'model_best.pth.tar'
save_path = checkpoint_path + 'model_best_single.pth.tar'
# print(checkpoint)
check_point = torch.load(load_path,map_location=lambda storage, loc: storage.cuda(0))
new_check_point = OrderedDict()
# for k, v in check_point.items():
#     print(k)
#
# for k, v in check_point['state_dict'].items():
#     print(k)
model = models.__dict__[args.path]()
model = model.cuda()

for k, v in check_point['state_dict'].items():
    # name = k[7:]  # remove `module.`
    print(k)
    name = k[9:]  # remove `module.1.`
    print(name)
    new_check_point[name] = v
# load params
model.load_state_dict(new_check_point)
torch.save(model.state_dict(), save_path)

# print(new_check_point)
