# Copyright (c) Midea Group
# Licensed under the Apache License 2.0

import os
import random
import sys
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from examples.utils.utils import AverageMeter, accuracy, model_arch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(20)

def train(model, device, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              .format(epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time))
        sys.stdout.flush()


def test(model, device, criterion, test_loader):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                idx, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for training')
    parser.add_argument('--model', type=str, default='resnet20', help='model help')
    args = parser.parse_args()

    # model_selection = "resnet56"
    # model_selection = "resnet50"
    model_selection = args.model
    data_path = '/data/cifar'
    model_path = '/pretrain/cifar100/mobilenet_v2/mobilenet_v2_cifar100.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset
    batch_size = 256
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = datasets.CIFAR100(root=data_path, train=True, transform=train_transform,
                                      download=False)
    test_dataset = datasets.CIFAR100(root=data_path, train=False, transform=test_transform,
                                     download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = model_arch(model_selection, 100, device)
    model.load_state_dict(state_dict=torch.load(model_path))

    criterion = torch.nn.CrossEntropyLoss()
    acc, test_acc_top5, test_loss = test(model, device, criterion, test_loader)
    print("acc:{}-top5acc:{}".format(acc, test_acc_top5))
