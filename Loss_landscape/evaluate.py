"""
nohup python evaluate.py > three_models.log &
"""
import argparse
import os
import random
import shutil
import time
import warnings
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import sys
sys.path.append('../')
import models as models
from load_pretrained import load_pretrained, rand_normalize_directions, get_combined_weights
import warnings
warnings.filterwarnings("ignore")


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='../data/imagenet/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    print("=> creating model 'old_resnet18, new1_resnet18, new3_resnet18'")
    model_old = models.__dict__["old_resnet18"]()
    model_new1 = models.__dict__["new1_resnet18"]()
    model_new3 = models.__dict__["new3_resnet18"]()



    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_old.cuda(args.gpu)
            model_new1.cuda(args.gpu)
            model_new3.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model_old = torch.nn.parallel.DistributedDataParallel(model_old, device_ids=[args.gpu])
            model_new1 = torch.nn.parallel.DistributedDataParallel(model_new1, device_ids=[args.gpu])
            model_new3 = torch.nn.parallel.DistributedDataParallel(model_new3, device_ids=[args.gpu])
        else:
            model_old.cuda()
            model_new1.cuda()
            model_new3.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_old = torch.nn.parallel.DistributedDataParallel(model_old)
            model_new1 = torch.nn.parallel.DistributedDataParallel(model_new1)
            model_new3 = torch.nn.parallel.DistributedDataParallel(model_new3)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_old = model_old.cuda(args.gpu)
        model_new1 = model_new1.cuda(args.gpu)
        model_new3 = model_new3.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_old = torch.nn.DataParallel(model_old).cuda()
        model_new1 = torch.nn.DataParallel(model_new1).cuda()
        model_new3 = torch.nn.DataParallel(model_new3).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

     # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # optionally resume from a checkpoint
    args.resume = 'old_resnet18_model_best.pth.tar'
    checkpoint_old = load_pretrained(args)
    direction1_old = rand_normalize_directions(checkpoint_old)
    direction2_old = rand_normalize_directions(checkpoint_old)
    args.resume = 'new1_resnet18_model_best.pth.tar'
    checkpoint_new1 = load_pretrained(args)
    direction1_new1 = rand_normalize_directions(checkpoint_new1)
    direction2_new1 = rand_normalize_directions(checkpoint_new1)
    args.resume = 'new3_resnet18_model_best.pth.tar'
    checkpoint_new3 = load_pretrained(args)
    direction1_new3 = rand_normalize_directions(checkpoint_new3)
    direction2_new3 = rand_normalize_directions(checkpoint_new3)


    print("=> loaded 3 combined checkpoint.")


    cudnn.benchmark = True

    # list_1 = np.arange(-0.5, 0.6, 0.1)
    # list_2 = np.arange(-0.5, 0.6, 0.1)
    list_1 = np.arange(-0.6, 0.6, 0.05)
    list_2 = np.arange(-0.6, 0.6, 0.05)

    print("initlizing logger")
    logger = logging.getLogger(args.arch)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler("old_new1_new3_out.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for w1 in list_1:
        for w2 in list_2:
            print("\n\n===> w1{w1:.2f} w2{w2:.2f}".format(w1=w1, w2=w2))
            combined_weights_old = get_combined_weights(direction1_old, direction2_old, checkpoint_old, w1,w2)
            model_old.load_state_dict(combined_weights_old)
            combined_weights_new1 = get_combined_weights(direction1_new1, direction2_new1, checkpoint_new1, w1,w2)
            model_new1.load_state_dict(combined_weights_new1)
            combined_weights_new3 = get_combined_weights(direction1_new3, direction2_new3, checkpoint_new3, w1,w2)
            model_new3.load_state_dict(combined_weights_new3)
            loss_old, loss_new1, loss_new3 = validate(val_loader, model_old, model_new1, model_new3, criterion, args)
            logger.info("{w1:.2f},{w2:.2f},{loss_old},{loss_new1},{loss_new3}".
                        format(w1=w1, w2=w2, loss_old=loss_old, loss_new1=loss_new1, loss_new3=loss_new3))


def validate(val_loader, model_old, model_new1, model_new3, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_old = AverageMeter('Loss', ':.4e')
    losses_new1 = AverageMeter('Loss', ':.4e')
    losses_new3 = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_old, losses_new1, losses_new3],
        prefix='Test: ')

    # switch to evaluate mode
    model_old.eval()
    model_new1.eval()
    model_new3.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model_old(images)
            loss_old = criterion(output, target)
            losses_old.update(loss_old.item(), images.size(0))

            output = model_new1(images)
            loss_new1 = criterion(output, target)
            losses_new1.update(loss_new1.item(), images.size(0))

            output = model_new3(images)
            loss_new3 = criterion(output, target)
            losses_new3.update(loss_new3.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(f"loss is {losses_old.avg} {losses_new1.avg} {losses_new3.avg}")


    return losses_old.avg, losses_new1.avg, losses_new3.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
    print("DONE, FINISHED!!!")
    os.system("sudo poweroff")
