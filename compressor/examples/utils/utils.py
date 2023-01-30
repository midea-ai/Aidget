# Copyright (c) Midea Group
# Licensed under the Apache License 2.0

import os
import torch
from torchvision.models import mobilenet_v2,shufflenet_v2_x0_5,shufflenet_v2_x1_0
from examples.model_zoo.resnet import resnet20, resnet56, resnet110
from examples.model_zoo.resnetv2 import resnet18,resnet34,resnet50,resnet101,resnet152

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        # print("self.last_epoch:{}".format(self.last_epoch))
        # print("self.total_iters:{}".format(self.total_iters))

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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



def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def model_arch(arch, num_cls, device):
    model = None
    if arch == "resnet50":
        model = resnet50(_num_classes=num_cls).to(device)
        print(" model is resnet50.")
    elif arch == "resnet18":
        model = resnet18(_num_classes=num_cls).to(device)
        print(" model is resnet18.")
    elif arch == "resnet34":
        model = resnet34(_num_classes=num_cls).to(device)
        print(" model is resnet34.")
    elif arch == "resnet20":
        model = resnet20(_num_classes=num_cls).to(device)
        print(" model is resnet20.")
    elif arch == "resnet56":
        model = resnet56(_num_classes=num_cls).to(device)
        print(" model is resnet56.")
    elif arch == "resnet110":
        model = resnet110(_num_classes=num_cls).to(device)
        print(" model is resnet110.")

    elif arch == "mobilenet_v2":
        model = mobilenet_v2(num_classes=num_cls).to(device)
        print(" model is mobilenet_v2.")
    elif arch == "shufflenet_v2_x0_5":
        model = shufflenet_v2_x0_5(num_classes=num_cls).to(device)
        print(" model is shufflenet_v2_x0_5.")
    elif arch == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x0_5(num_classes=num_cls).to(device)
        print(" model is shufflenet_v2_x1_0.")
    else:
        raise NotImplementedError(arch)
    return model
