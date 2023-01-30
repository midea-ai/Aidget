# Copyright (c) Midea Group
# Licensed under the Apache License 2.0


import os
import random
import sys
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from examples.utils.utils import AverageMeter, accuracy, WarmUpLR, create_dir
import argparse
from utils.utils import model_arch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(20)

def train(model, device, train_loader, criterion, optimizer, epoch, args):
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

        if epoch < args.warm:
            warmup_scheduler.step()


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
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for training')
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--warm', type=int, default=5, help='warm up training phase, set 0 to disable warm-up')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--gamma', type=float, default=0.2, help='decay factor')
    parser.add_argument('--eps', type=int, default=240, help='total epochs')
    parser.add_argument('--ms', nargs='+', type=int, default=[70, 140, 190], help='milestones')

    args = parser.parse_args()

    batch_size = args.bs
    LR = 0.1
    epochs = args.eps
    num_classes = 100
    model_selection = args.model
    save_dir = '/pretrain/cifar100/{}'.format(model_selection)
    data_path = '/data/cifar'
    create_dir(save_dir)
    # dataset
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
    model = model_arch(model_selection, num_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    print("epochs:{} milestones:{} warmup:{} gamma:{} batch size:{}".format(epochs, args.ms, args.warm, args.gamma,
                                                                            batch_size))
    scheduler = MultiStepLR(optimizer, milestones=args.ms, gamma=args.gamma)

    warmup_scheduler = None
    if args.warm > 0:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    print('start training...')
    acc, test_acc_top5, test_loss = test(model, device, criterion, test_loader)
    best_acc = acc
    best_state_dict = None
    for epoch in range(epochs):
        train(model, device, train_loader, criterion, optimizer, epoch, args)
        if epoch >= args.warm:
            scheduler.step()
        acc, test_acc_top5, test_loss = test(model, device, criterion, test_loader)
        if acc > best_acc:
            best_ep = epoch
            best_acc = acc
            best_acc_top5 = test_acc_top5
            save_path = os.path.join(save_dir, "{}_cifar100_epoch_{}.pt".format(model_selection, epochs))
            torch.save(model.state_dict(), save_path)
            print("model saved at {}".format(save_path))
    print("best saved")
    print("best model epoch:{}".format(best_ep))
    model.load_state_dict(torch.load(save_path))
    acc, test_acc_top5, test_loss = test(model, device, criterion, test_loader)
