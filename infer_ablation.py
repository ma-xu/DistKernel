"""
ImageNet training script.
Including APEX (distributed training), and DALI(data pre-processing using CPU+GPU) provided by NIVIDIA.
Thanks pytorch demo, implus (Xiang Li from NJUST), DALI.
Author: Xu Ma
Date: Aug/15/2019
Email: xuma@my.unt.edu

Useage:
python3 -m torch.distributed.launch --nproc_per_node=8 main -a old_resnet50 --fp16 --b 32



"""


import argparse
import os
import shutil
import time
import traceback
from collections import OrderedDict

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models as models
from utils import Logger, mkdir_p


try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/home/g1007540910/DATA/ImageNet2012/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='new1_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='/home/g1007540910/DistKernel/checkpoints/imagenet/new1_resnet18/model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--rm', nargs='+', help='Layers need to be reset as zero')

parser.add_argument("--local_rank", default=0, type=int)

cudnn.benchmark = True

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

best_prec1 = 0
args = parser.parse_args()


args.distributed = False

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    global best_prec1, args

    args.gpu = 0
    args.world_size = 1

    args.total_batch_size = args.world_size * args.batch_size


    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.rm:
        for layer in args.rm:
            print("Remove branch: {}".format(layer))
    else:
        print("Keep all branches")



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                # print(k)
                # name = k[9:]  # remove `module.1.`
                # print(name)
                for layer in args.rm:
                    if layer in k:
                        v = v-v # set to zero but keep the same shape
                        print(f"Setting layer\t {k} \t paprameters to zero.")

                state_dict[name] = v
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))





    # traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    crop_size = 224
    val_size = 256

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))


        reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Speed {2:.3f} \t'
                  'Loss {loss.avg:.4f}\t'
                  'Top1 {top1.avg:.3f}\t'
                  'Top5 {top5.avg:.3f}'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    if args.local_rank == 0:
        print(' TEST Top1 {top1.avg:.4f} Top5 {top5.avg:.4f}'.format(top1=top1, top5=top5))

    return [losses.avg, top1.avg,top5.avg]



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
    #     os.system("sudo poweroff")
    # print("DONE, FINISHED!!!")
    # os.system("sudo poweroff")
    main()
