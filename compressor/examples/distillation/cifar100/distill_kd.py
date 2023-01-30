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
from mslim.algorithms.pytorch.distillation.distiller import Distiller
from examples.utils.utils import AverageMeter, accuracy, model_arch, WarmUpLR
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


def train(modelS, device, train_loader, criterion, optimizer, epoch, distiller):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    modelS.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred, kd_loss = distiller.forward(data)
        output = pred
        loss = criterion(output, target)
        losses = loss * 0.8 + kd_loss
        losses.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        # warm up
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
    parser.add_argument('--model_t', type=str, default='resnet56', help='teacher model architecture')
    parser.add_argument('--model_s', type=str, default='resnet20', help='student model architecture')
    parser.add_argument('--warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    args = parser.parse_args()

    # save_dir = "models/modelzoo/"
    teacher_model_selection = args.model_t
    student_model_selection = args.model_s
    distill_method = 'vanilla_kd'
    save_dir = '/distill/cifar100/{}_{}_{}'.format(
        teacher_model_selection, student_model_selection, distill_method)
    data_path = '/data/cifar'

    num_cls = 100
    LR = 0.01
    epochs = 200
    batch_size = args.bs
    print("teacher model selection")
    teacher = model_arch(teacher_model_selection, num_cls, device)
    print("student model selection")
    distilled_student = model_arch(student_model_selection, num_cls, device)

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

    modelS = distilled_student
    modelT = teacher

    modelT.load_state_dict(torch.load(os.path.join('/pretrain'
                                                   '/cifar100/{}'.format(teacher_model_selection),
                                                   "{}_cifar100.pt".format(teacher_model_selection))))
    modelS.load_state_dict(torch.load(os.path.join('/pretrain/cifar100/{}'.format(student_model_selection),
                                                   "{}_cifar100.pt".format(student_model_selection))))

    config = {
        'student_model': modelS,
        'teacher_model': modelT,
        'student_distill_layer': ['linear'],
        'teacher_distill_layer': ['linear'],
        'loss': [dict(type='vanilla_kd', T=4)],
        'weights': [0.2],
        'student_out_layer': ['linear'],
        'device': device
    }

    distiller_instance = Distiller(config)
    criterion = torch.nn.CrossEntropyLoss()
    acc, test_acc_top5, test_loss = test(modelT, device, criterion, test_loader)
    print("teacher acc:@1:{} @5:{}".format(acc, test_acc_top5))
    optimizer = torch.optim.SGD(distiller_instance.get_learnable_params(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160],
                            gamma=0.2)
    iter_per_epoch = len(train_loader)

    # warm up
    warmup_scheduler = None
    if args.warm > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    print('start training...')
    best_acc = 0
    best_state_dict = None
    for epoch in range(epochs):
        print("epoch:{}/{}".format(epoch, epochs))
        for param_group in optimizer.param_groups:
            print("lr:{}".format(param_group['lr']))
        train(modelS, device, train_loader, criterion, optimizer, epoch, distiller_instance)
        if epoch >= args.warm:
            scheduler.step()
        acc, test_acc_top5, test_loss = test(modelS, device, criterion, test_loader)
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(save_dir, "{}_{}_{}_cifar100_epoch_{}.pt".format(teacher_model_selection,
                                                                                      student_model_selection,
                                                                                      distill_method, epochs))
            torch.save(modelS.state_dict(), save_path)
            print("model saved at {}".format(save_path))
    print("best acc:@1:{}".format(best_acc))
