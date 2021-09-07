import warnings
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from collections import OrderedDict
import models as models

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-d', '--data', default='/home/g1007540910/DATA/ImageNet2012/', type=str)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='old_resnet18')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='/Users/xuma/Downloads/old_resnet18_model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')

    parser.add_argument("--local_rank", default=0, type=int)
    return parser.parse_args()

def load_pretrained(args):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    new_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        # if k.startswith("module.1."):
        #     k = k[9:]
        # if k.startswith("module."):
        #     k = k[7:]
        if k.startswith("module.1."):
            k[0:9] = k[0:7]
        new_dict[k] = v
    return new_dict


def rand_normalize_directions(args, states, ignore='ignore'):
    # assert(len(direction) == len(states))
    model = models.__dict__[args.arch]()
    init_dict = model.state_dict()
    new_dict = OrderedDict()
    for (k, w), (k2, d) in zip(states.items(), init_dict.items()):
        # if w.dim() <= 1:
        #     if ignore == 'biasbn':
        #         d = torch.zeros_like(w)  # ignore directions for weights with 1 dimension
        #     else:
        #         d = w
        # else:
        #     d.mul_(w.norm()/(d.norm() + 1e-10))
        new_dict[k] = d
    return new_dict


def get_combined_weights(direction1, direction2, pretrained, weight1, weight2, weight_pretrained=1.0):
    new_dict = OrderedDict()
    for (k, d1),(_,d2), (_,w) in zip(direction1.items(), direction2.items(), pretrained.items()):
        new_dict[k] = (weight1 * d1 + weight2 * d2 + weight_pretrained * w)/(weight1+weight2+weight_pretrained)
    return new_dict


if __name__ == '__main__':
    args = get_parser()
    pretrained_weights = load_pretrained(args)
    direction1 = rand_normalize_directions(args, pretrained_weights)
    direction2 = rand_normalize_directions(args, pretrained_weights)
    # import numpy as np
    # list_1 = np.arange(-1, 1.1, 0.1)
    # print(list_1)
    # combined = get_combined_weights(direction1, direction2, pretrained_weights, 1,1)
    # print(combined)
