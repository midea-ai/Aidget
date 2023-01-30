# Copyright (c) Midea Group
# Licensed under the Apache License 2.0


import torch
from torch import nn
from .loss import loss_dict
from .loss_zoo import IRGLoss
from .utils import get_module_by_name


class Distiller(nn.Module):
    def __init__(self, config, **kwargs):
        super(Distiller, self).__init__()
        self.student_net = config['student_model']
        self.teacher_net = config['teacher_model']
        self.student_distill_layer = config['student_distill_layer']
        self.teacher_distill_layer = config['teacher_distill_layer']
        self.student_out_layer = config['student_out_layer']
        self.loss_list = config['loss']
        self.losses = []
        self.weights = config['weights']
        self.device = config['device']
        self.learnable_list = nn.ModuleList()
        self.student_output_module = []
        self.teacher_output_module = []
        self.s_features_out_hook = {}
        self.t_features_out_hook = {}
        self.out_module = None
        self.build()

    def show_hook_t(self):
        print("show_hook_teacher")
        for k, v in self.t_features_out_hook.items():
            print(k)

    def show_hook_s(self):
        print("show_hook_student")
        for k, v in self.s_features_out_hook.items():
            print(k)

    def student_forward_output_hook(self, module, inputs, fea_out):
        # DDP
        self.s_features_out_hook[module].append(fea_out)
        return None

    def teacher_forward_output_hook(self, module, inputs, fea_out):
        # DDP
        self.t_features_out_hook[module].append(fea_out)
        return None

    def build_hook(self):
        # student
        s_net = self.student_net
        for name, module in s_net.named_modules():
            if module == s_net:
                continue
            if name in self.student_distill_layer:
                print("student layer {} added".format(name))
                self.student_output_module.append(get_module_by_name(s_net, name))

        # teacher
        t_net = self.teacher_net
        for name, module in t_net.named_modules():
            if module == t_net:
                continue
            if name in self.teacher_distill_layer:
                print("teacher layer {} added".format(name))
                self.teacher_output_module.append(get_module_by_name(t_net, name))

        for module in self.student_output_module:
            self.s_features_out_hook[module] = []
            module.register_forward_hook(self.student_forward_output_hook)

        for module in self.teacher_output_module:
            self.t_features_out_hook[module] = []
            module.register_forward_hook(self.teacher_forward_output_hook)

        # student output hook, currently only support 1 output module.
        self.out_module = get_module_by_name(s_net, self.student_out_layer[0])
        print("debug- out module:{}".format(self.out_module))
        if self.out_module in self.student_output_module:
            print("Layer {} is already added".format(self.student_out_layer[0]))
        else:
            print("Layer {} is added for output".format(self.student_out_layer[0]))
            self.s_features_out_hook[self.out_module] = []
            self.out_module.register_forward_hook(self.student_forward_output_hook)

    def build_losses(self):
        """build losses."""
        for loss in self.loss_list:
            self.losses.append(loss_dict[loss['type']](**loss))
        print("Build losses:{}".format(self.losses))

    def build(self):
        """build."""
        self.build_hook()
        self.build_losses()
        self.build_learnable_list()

    def clear_outputs(self, outputs):
        """Reset the outputs."""
        for key in outputs.keys():
            outputs[key] = list()

    def forward_student(self, data):
        """student net inference."""
        self.clear_outputs(self.s_features_out_hook)
        output = self.student_net(data)
        return output

    def forward_teacher(self, data):
        """teacher net inference."""
        self.clear_outputs(self.t_features_out_hook)
        with torch.no_grad():
            output = self.teacher_net(data)
        return output

    def forward(self, data):
        """distiller forward."""
        losses = None
        self.forward_student(data)
        self.forward_teacher(data)
        # log only
        loss_values = []
        if type(self.losses[0]) is IRGLoss:
            feat_s = nn.AvgPool2d(8)(self.s_features_out_hook[self.student_output_module[1]][0])
            feat_s = feat_s.view(feat_s.size(0), -1)
            feat_t = nn.AvgPool2d(8)(self.t_features_out_hook[self.teacher_output_module[1]][0])
            feat_t = feat_t.view(feat_t.size(0), -1)
            tmp_loss = self.losses[0]([self.s_features_out_hook[self.student_output_module[0]][0],
                                       self.s_features_out_hook[self.student_output_module[1]][0],
                                       feat_s,
                                       self.s_features_out_hook[self.student_output_module[2]][0]],
                                      [self.t_features_out_hook[self.teacher_output_module[0]][0],
                                       self.t_features_out_hook[self.teacher_output_module[1]][0],
                                       feat_t,
                                       self.t_features_out_hook[self.teacher_output_module[2]][0]]
                                      )
            loss_values.append(tmp_loss)
            losses = tmp_loss
        else:
            for i, (loss, fea_s_module, fea_t_module) in enumerate(
                    zip(self.losses, self.student_output_module, self.teacher_output_module)):
                fea_s = self.s_features_out_hook[fea_s_module][0]
                fea_t = self.t_features_out_hook[fea_t_module][0]
                tmp_loss = loss(fea_s, fea_t) * self.weights[i]
                loss_values.append(tmp_loss)
                if i == 0:
                    losses = tmp_loss
                else:
                    losses += tmp_loss

        return self.s_features_out_hook[self.out_module][0], losses

    def get_learnable_params(self):
        """return learnable parameters."""
        return self.learnable_list.parameters()

    def build_learnable_list(self):
        """build learnable items."""
        self.learnable_list.append(self.student_net)
        for loss in self.losses:
            self.learnable_list.append(loss)
        self.learnable_list.to(self.device)

    def show_student_net(self):
        """show student net."""
        print("showing student net:\n")
        net = self.student_net
        for name, module in net.named_modules():
            if module == net:
                continue
            print("name:{}".format(name))
            print("module:{}".format(module))

    def show_teacher_net(self):
        """show teacher net."""
        print("showing teacher net:\n")
        net = self.teacher_net
        for name, module in net.named_modules():
            if module == net:
                continue
            print("name:{}".format(name))
            print("module:{}".format(module))