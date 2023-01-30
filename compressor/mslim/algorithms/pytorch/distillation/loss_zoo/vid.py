# Copyright (c) Midea Group
# Licensed under the Apache License 2.0

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation
    code is modified from https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/vid.py
    """

    def __init__(self,
                 **dict):
        super(VIDLoss, self).__init__()
        self.inC = dict['inC']
        self.midC = dict['midC']
        self.OutC = dict['outC']
        self.init_pred_var = dict['init_pred_var']
        self.eps = dict['eps']

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(self.inC, self.midC),
            nn.ReLU(),
            conv1x1(self.midC, self.midC),
            nn.ReLU(),
            conv1x1(self.midC, self.OutC),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(self.init_pred_var - self.eps) - 1.0) * torch.ones(self.OutC)
        )

    def forward(self, input, target):
        # pool for dimension match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
                (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )
        loss = torch.mean(neg_log_prob)
        return loss
