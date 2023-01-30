# Copyright (c) Midea Group
# Licensed under the Apache License 2.0
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv7KDLoss(nn.Module):
    """
    Distill the output's knowledge for YOLOv7.
    The code is modified from https://github.com/Tyler-Cheen/Knowledge-Distillation-yolov5/blob/main/loss.py
    """

    def __init__(self, **dict):
        super(YOLOv7KDLoss, self).__init__()
        print("dict:{}".format(dict))
        # self.h = dict['h'] #h = model.hyp
        self.w_giou = dict['giou']
        self.w_dist = dict['dist']
        self.w_obj = dict['obj']
        self.w_cls = dict['cls']
        self.nc = dict['nc']

    def forward(self, p, t_p):
        t_p = t_p[1]
        t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
        t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
        red = 'mean'
        if red != "mean":
            raise NotImplementedError("reduction must be mean in distillation mode!")
        DboxLoss = nn.MSELoss(reduction="none")
        DclsLoss = nn.MSELoss(reduction="none")
        DobjLoss = nn.MSELoss(reduction="none")
        for i, pi in enumerate(p):
            t_pi = t_p[i]
            t_obj_scale = t_pi[..., 4].sigmoid()
            b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
            t_lbox += torch.mean(DboxLoss(pi[..., :4], t_pi[..., :4]) * b_obj_scale)
            if self.nc > 1:
                c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, self.nc)
                t_lcls += torch.mean(DclsLoss(pi[..., 5:], t_pi[..., 5:]) * c_obj_scale)
            t_lobj += torch.mean(DobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
        t_lbox *= self.w_giou * self.w_dist
        t_lobj *= self.w_obj * self.w_dist
        t_lcls *= self.w_cls * self.w_dist
        bs = p[0].shape[0]
        loss = (t_lobj + t_lbox + t_lcls) * bs
        return loss
