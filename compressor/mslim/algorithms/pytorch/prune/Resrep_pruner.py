# Copyright (c) Midea Group
# Licensed under the Apache License 2.0

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.fx import symbolic_trace

import copy
import time
import logging
from collections import defaultdict
import numpy as np


class CompactorLayer(torch.nn.Module):

    def __init__(self, num_features):
        super(CompactorLayer, self).__init__()
        self.pwc = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1,
                          stride=1, padding=0, bias=False)
        identity_mat = np.eye(num_features, dtype=np.float32)
        self.pwc.weight.data.copy_(torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1))
        self.register_buffer('mask', torch.ones(num_features))
        init.ones_(self.mask)
        self.num_features = num_features

    def forward(self, inputs):
        return self.pwc(inputs)

    def set_mask(self, zero_indices):
        new_mask_value = np.ones(self.num_features, dtype=np.float32)
        new_mask_value[np.array(zero_indices)] = 0.0
        self.mask.data = torch.from_numpy(new_mask_value).cuda().type(torch.cuda.FloatTensor)

    def set_weight_zero(self, zero_indices):
        new_compactor_value = self.pwc.weight.data.detach().cpu().numpy()
        new_compactor_value[np.array(zero_indices), :, :, :] = 0.0
        self.pwc.weight.data = torch.from_numpy(new_compactor_value).cuda().type(torch.cuda.FloatTensor)

    def get_num_mask_ones(self):
        mask_value = self.mask.cpu().numpy()
        return np.sum(mask_value == 1)

    def get_remain_ratio(self):
        return self.get_num_mask_ones() / self.num_features

    def get_pwc_kernel_detach(self):
        return self.pwc.weight.detach()

    def get_lasso_vector(self):
        lasso_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return lasso_vector

    def get_metric_vector(self):
        metric_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return metric_vector


class InsertCompactor(nn.Module):
    def __init__(self, module):
        super(InsertCompactor, self).__init__()
        self.module = module
        assert isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm)
        self.compactor = CompactorLayer(self.module.num_features)
        return

    def forward(self, x):
        x = self.module(x)
        x = self.compactor(x)
        return x


class ResRep:
    def __init__(self, model):
        self.model = model
        self.config = []
        self.bn_list = []
        self.conv_list = []
        self.target_module = []
        self.exclude = set()
        self.residual_conv = set()
        self.bn_conv_dict = dict()
        self.conv_bn_dict = dict()
        self.target_module_dict = dict()
        self.original_layer_ones_dict = dict()
        self.connections = defaultdict(list)
        self.concat_relations = defaultdict(list)
        self.reversed_connections = defaultdict(list)

        self.generate_bn_conv_dict()
        self.prepare_tricky_layers()
        self.configure()

        self.optimizer = None
        self.wrapped_model = None

        self.replace_module()
        self.get_compactor_mask_score()
        self.find_target_module()
        self.generate_target_module_dict()

    def generate_bn_conv_dict(self):
        for name, layer in self.model.named_modules():
            name = 'module.' + name
            if isinstance(layer, nn.Conv2d):
                self.conv_list.append(name)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm):
                self.bn_list.append(name)
        for idx, bn in enumerate(self.bn_list):
            self.bn_conv_dict[bn] = self.conv_list[idx]
            self.conv_bn_dict[self.conv_list[idx]] = bn
        return self.bn_conv_dict
    
    def configure(self):

        for name, layer in self.model.named_modules():
            if not name.startswith('module.'):
                name = 'module.' + name

            if (isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm)):

                if len(self.connections[self.bn_conv_dict[name].replace('module.', '')]) == 0:
                    if self.bn_conv_dict[name] not in self.conv_list:
                        self.exclude.add(self.bn_conv_dict[name].replace('module.', ''))
                    else:
                        if self.bn_conv_dict[name].replace('module.', '') not in self.residual_conv:
                            self.config.append(name)

                elif self.connections[self.bn_conv_dict[name].replace('module.', '')][0] not in self.connections:

                    if ('module.' + self.connections[self.bn_conv_dict[name].replace('module.', '')][0]) not in self.conv_bn_dict:
                        self.exclude.add(self.bn_conv_dict[name].replace('module.', ''))

                    if 'rbr' in name:
                        for conv in self.reversed_connections[self.bn_conv_dict[name].replace('module.', '')]:
                            self.exclude.add(conv)
                    else:
                        if self.bn_conv_dict[name].replace('module.', '') not in self.residual_conv:
                            self.config.append(name)

                else:
                    if self.bn_conv_dict[name].replace('module.', '') not in self.residual_conv:
                        self.config.append(name)

        for exclude_conv in self.exclude:
            if self.conv_bn_dict['module.' + exclude_conv] in self.config:
                self.config.remove(self.conv_bn_dict['module.' + exclude_conv])

        return self.config

    def prepare_tricky_layers(self):
        traced = symbolic_trace(self.model)  # augment=False, profile=False
        for node in traced.graph.nodes:
            if node.op == "call_method":
                continue
            if len(node.args) > 1 and 'add' in str(node):
                for i in range(len(node.args)):
                    new_node = node.args[i]
                    if isinstance(new_node, int):
                        continue
                    if new_node.op == 'call_function':
                        continue
                    module = visit_module(self.model, new_node.target)
                    while not isinstance(module, nn.Conv2d):
                        flag = 0
                        for node2 in new_node.args:
                            if node2.op == "call_function":
                                continue
                            else:
                                flag = 1
                                new_node = node2
                                module = visit_module(self.model, new_node.target)
                                if isinstance(module, nn.Conv2d):
                                    self.residual_conv.add(new_node.target)
                        if flag == 0:
                            break
        self.residual_conv = sorted(self.residual_conv)

        for node in traced.graph.nodes:
            if 'cat' in str(node):
                concat_tail = None
                for key in node.users.keys():
                    concat_tail = key
                    assert isinstance(visit_module(self.model, key.target), nn.Conv2d)

                for head in node.args[0]:
                    module = visit_module(self.model, head.target)
                    while not isinstance(module, nn.Conv2d):
                        if len(head.args) == 1:
                            node_name = head.args[0]
                            node_target = node_name.target
                            module = visit_module(self.model, node_target)
                            if isinstance(module, nn.Conv2d):
                                self.concat_relations[str(concat_tail.target)].append(node_target)
                            head = node_name
                        else:
                            raise NotImplementedError

        for node in traced.graph.nodes:
            if node.op != "call_module":
                continue
            if isinstance(visit_module(self.model, node.target), nn.Conv2d):
                if node.target == self.conv_list[0]:
                    continue
                new_node = node.args[0]
                if new_node.op == 'call_module':
                    module = visit_module(self.model, new_node.target)
                    while not isinstance(module, nn.Conv2d):
                        if len(new_node.args) == 1:
                            new_node = new_node.args[0]
                            if new_node.op == 'call_function':
                                for i in range(len(new_node.args)):
                                    new_node_2 = new_node.args[i]
                                    if isinstance(new_node_2, int):
                                        continue
                                    while new_node_2.op == 'call_function':
                                        flag = 0
                                        for j in range(len(new_node_2.args)):
                                            new_node_3 = new_node_2.args[j]
                                            module = visit_module(self.model, new_node_3.target)
                                            while not isinstance(module, nn.Conv2d):
                                                new_node_4 = new_node_3.args[0]
                                                module = visit_module(self.model, new_node_4.target)
                                                if isinstance(module, nn.Conv2d):
                                                    flag += 1
                                                    self.reversed_connections[str(node.target)].append(new_node_4.target)
                                        if flag == len(new_node_2.args):
                                            break
                            else:
                                module = visit_module(self.model, new_node.target)
                                if isinstance(module, nn.Conv2d):
                                    self.reversed_connections[str(node.target)].append(new_node.target)
                        else:
                            for j in range(len(new_node.args)):
                                new_node_3 = new_node.args[j]
                                if new_node_3.op == 'call_function':
                                    continue
                                module = visit_module(self.model, new_node_3.target)
                                while not isinstance(module, nn.Conv2d):
                                    new_node_3 = new_node_3.args[0]
                                    if new_node_3.op == 'call_function':
                                        continue
                                    module = visit_module(self.model, new_node_3.target)
                                    if isinstance(module, nn.Conv2d):
                                        self.reversed_connections[str(node.target)].append(new_node_3.target)
                else:
                    if 'cat' in str(new_node):
                        concat_heads = new_node.args[0]
                        for concat_head in concat_heads:
                            new_node = concat_head.args[0]
                            module = visit_module(self.model, new_node.target)
                            while not isinstance(module, nn.Conv2d):
                                if len(new_node.args) == 1:
                                    new_node = new_node.args[0]
                                    module = visit_module(self.model, new_node.target)
                                    if isinstance(module, nn.Conv2d):
                                        self.reversed_connections[str(node.target)].append(new_node.target)
                                else:
                                    raise NotImplementedError

        for key in self.reversed_connections.keys():
            for val in self.reversed_connections[key]:
                self.connections[val].append(key)

    def replace_module(self):
        for module_name in self.config:
            if not hasattr(self.model, 'module'):
                module_name = module_name.replace('module.', '')
            target_module = visit_module(self.model, module_name)
            self.wrapped_model = self.reset_module(self.model, module_name, InsertCompactor(target_module))
        return self.wrapped_model

    def reset_module(self, original_model, module_name, wrapper):
        self.wrapped_model = original_model
        seperated_name = module_name.split(".")
        cur_name = ".".join(seperated_name[:-1])
        higher_attr = visit_module(self.wrapped_model, cur_name) if len(cur_name) != 0 else self.wrapped_model
        setattr(higher_attr, seperated_name[-1], wrapper)
        self.wrapped_model.eval()
        return self.wrapped_model

    def find_target_module(self):
        index = 0
        index_dict = dict()
        for name, layer in self.wrapped_model.named_modules():
            index += 1
            index_dict[index] = name
            if type(layer).__name__ == 'InsertCompactor':
                self.target_module.append([index_dict[index - 1], name + '.module', name + '.compactor'])

    def generate_target_module_dict(self):
        for idx in range(len(self.target_module)):
            self.target_module_dict[self.target_module[idx][0]] = self.target_module[idx][2]

    def sgd_optimizer(self, momentum):
        assert momentum, f'momentum not assigned yet!'
        params = []
        lr = 0.01
        for key, value in self.wrapped_model.named_parameters():
            if not value.requires_grad:
                continue
            weight_decay = 0 if 'bias' in key or 'bn' in key else 1e-4
            apply_lr = 2 * lr if 'bias' in key else lr
            if 'compactor' in key:
                use_momentum = 0.99
            else:
                use_momentum = momentum
            params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay, "momentum": use_momentum}]
        self.optimizer = torch.optim.SGD(params, lr, momentum=momentum)
        return self.optimizer

    def get_compactor_mask_score(self):
        for name, layer in self.wrapped_model.named_modules():
            if not name.startswith('module.'):
                name = 'module.' + name
            if type(layer).__name__ == 'InsertCompactor':
                conv_name = self.bn_conv_dict[name]
                self.original_layer_ones_dict[conv_name] = layer.compactor.get_num_mask_ones()
        return self.original_layer_ones_dict


def visit_module(model, module_name, flag=0):
    module_name = module_name.replace('module.', '')
    if flag == 1:
        seperated_name = ['module'] + module_name.split(".")
    else:
        seperated_name = module_name.split(".")
    forward = model
    for i in range(len(seperated_name)):
        name = seperated_name[i]
        if name in ["contiguous", "view", "sigmoid", "permute"]:  # patch
            continue
        forward = getattr(forward, name)
        if isinstance(forward, nn.Identity):
            break
    return forward


def calculate_model_flops(modified_model, dummy_input, ResRep_object, flag=None):
    layer_flops_dict = defaultdict(list)
    handle_list = []
    for name, layer in modified_model.named_modules():
        if (isinstance(layer, nn.Conv2d) and 'pwc' not in name) or isinstance(layer, nn.Linear):
            handle = layer.register_forward_hook(versatile(name, layer_flops_dict, modified_model, ResRep_object, flag))
            handle_list.append(handle)
    if flag == 1:
        modified_model(dummy_input)
    else:
        modified_model.float().cuda()(dummy_input)
    for handle in handle_list:
        handle.remove()
    return flops


def versatile(name, layer_flops_dict, modified_model, ResRep_object, flag):
    def forward_hook(module, data_input, data_output):
        global flops
        if isinstance(module, nn.Conv2d):
            layer_flops_dict[name.replace('module.', '')] = [data_output[0].shape[1], data_output[0].shape[2],
                                      module.kernel_size[0], module.kernel_size[1], module.groups]
            if name in ResRep_object.residual_conv:
                layer_flops_dict[name.replace('module.', '')].append(module.out_channels)
            else:
                if flag == 1:
                    if name.replace('module.', '') in ResRep_object.target_module_dict:
                        succeeding_compactor = visit_module(modified_model, ResRep_object.target_module_dict[name.replace('module.', '')], 1)
                        layer_flops_dict[name.replace('module.', '')].append(succeeding_compactor.get_num_mask_ones())
                    else:
                        layer_flops_dict[name.replace('module.', '')].append(module.out_channels)
                else:
                    layer_flops_dict[name.replace('module.', '')].append(module.out_channels)
            if len(ResRep_object.reversed_connections[name.replace('module.', '')]) == 0:
                layer_flops_dict[name.replace('module.', '')].append(3)
            elif len(ResRep_object.reversed_connections[name.replace('module.', '')]) > 1:
                sum = 0
                for concat_head in ResRep_object.reversed_connections[name.replace('module.', '')]:
                    sum += layer_flops_dict[concat_head][5]
                layer_flops_dict[name.replace('module.', '')].append(sum)
            else:
                for key2 in layer_flops_dict.keys():
                    if ResRep_object.reversed_connections[name.replace('module.', '')][0] == key2:
                        layer_flops_dict[name.replace('module.', '')].append(layer_flops_dict[key2][5])
                        break
        if isinstance(module, nn.Linear):
            layer_flops_dict[name.replace('module.', '')] = [module.out_features * module.in_features]
        flops = calculate_flops(layer_flops_dict)

    return forward_hook


def calculate_flops(layer_flops_dict):
    flops = 0
    for layer in layer_flops_dict.keys():
        if len(layer_flops_dict[layer]) == 1:
            flops += layer_flops_dict[layer][0]
        else:
            h, w, kernel_size_0, kernel_size_1, groups, out_channel, in_channel = layer_flops_dict[layer][0], \
                                                                                  layer_flops_dict[layer][1], \
                                                                                  layer_flops_dict[layer][2], \
                                                                                  layer_flops_dict[layer][3], \
                                                                                  layer_flops_dict[layer][4], \
                                                                                  layer_flops_dict[layer][5], \
                                                                                  layer_flops_dict[layer][6]
            if out_channel == 0:
                print('****', layer)
                print("Warning: have been decreased to 0 !")
                out_channel = 1
            flops += h * w * kernel_size_0 * kernel_size_1 * in_channel * out_channel // groups
    return flops


def get_compactor_mask_score(model, ResRep_object):
    layer_mask_ones = dict()
    layer_metric_dict = dict()
    for name, layer in model.named_modules():
        if not name.startswith('module.'):
            name = 'module.' + name
        if type(layer).__name__ == 'InsertCompactor':
            conv_name = ResRep_object.bn_conv_dict[name]
            layer_mask_ones[conv_name] = layer.compactor.get_num_mask_ones()
            metric_vector = layer.compactor.get_metric_vector()
            for i in range(len(metric_vector)):
                layer_metric_dict[(conv_name, i)] = metric_vector[i]
    return layer_mask_ones, layer_metric_dict


def get_compactor_mask_dict(model):
    compactor_name_to_mask = dict()
    compactor_name_to_kernel_param = dict()
    for name, buffer in model.named_buffers():
        if 'compactor.mask' in name:
            compactor_name_to_mask[name.replace('mask', '')] = buffer
    for name, param in model.named_parameters():
        if 'compactor.pwc.weight' in name:
            compactor_name_to_kernel_param[name.replace('pwc.weight', '')] = param
    result = dict()
    for name, kernel in compactor_name_to_kernel_param.items():
        mask = compactor_name_to_mask[name]
        num_filters = mask.nelement()
        if kernel.ndimension() == 4 and mask.ndimension() == 1:
            broadcast_mask = mask.reshape(-1, 1).repeat(1, num_filters)
            result[kernel] = broadcast_mask.reshape(num_filters, num_filters, 1, 1)
    return result


def compute_deactivated(model, origin_layer_ones_dict, cur_layer_ones_dict):
    result = 0
    for key in origin_layer_ones_dict.keys():
        result += origin_layer_ones_dict[key] - cur_layer_ones_dict[key]
        if origin_layer_ones_dict[key] - cur_layer_ones_dict[key] > 0:
            logging.debug(f'{key} : {origin_layer_ones_dict[key] - cur_layer_ones_dict[key]}')
    return result


def set_model_masks(model, layer_masked_out_filters, ResRep_object):
    for name, layer in model.named_modules():
        if type(layer).__name__ == 'InsertCompactor':
            if not name.startswith('module.'):
                name = 'module.' + name
            conv_name = ResRep_object.bn_conv_dict[name]
            if conv_name in layer_masked_out_filters:
                layer.compactor.set_mask(layer_masked_out_filters[conv_name])


def sgd_optimizer(model):
    params = []
    lr = 0.01
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = 0 if 'bias' in key or 'bn' in key else 1e-4
        apply_lr = 2 * lr if 'bias' in key else lr
        if 'compactor' in key:
            use_momentum = 0.99
        else:
            use_momentum = 0.9
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay, "momentum": use_momentum}]
    optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    return optimizer


def fuse_conv_bn(conv, bn):
    if isinstance(bn, nn.Identity):
        return conv
    else:
        fused_conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels,
                               kernel_size=conv.kernel_size, stride=conv.stride, groups=conv.groups,
                               padding=conv.padding, bias=True).requires_grad_(False).to(conv.weight.device)
        conv_weight = conv.weight.clone().reshape(conv.out_channels, -1)
        bn_weight = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var))) if hasattr(bn, 'weight') else torch.diag(bn.module.weight.div(torch.sqrt(bn.module.eps+bn.module.running_var)))
        bn_bias = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.eps + bn.running_var)) if hasattr(bn, 'weight') else bn.module.bias - bn.module.weight.mul(bn.module.running_mean).div(torch.sqrt(bn.module.eps + bn.module.running_var))

        fused_conv.weight.copy_(torch.matmul(bn_weight, conv_weight).reshape(fused_conv.weight.shape))
        if conv.bias is not None:
            conv_bias = conv.bias
        else:
            conv_bias = torch.zeros(conv.weight.size(0)).to(conv.weight.device)
        conv_bias = torch.mm(bn_weight, conv_bias.reshape(-1, 1)).reshape(-1)
        fused_conv.bias.copy_(conv_bias + bn_bias)
        return fused_conv


def set_module(model, module_name, module):
    seperated_names = module_name.split('.')
    super_name = seperated_names[:-1]
    cur_mod = model
    for s in super_name:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, seperated_names[-1], module)


def Resrep_2(opt, compactor_mask_dict):
    for compactor_param, mask in compactor_mask_dict.items():
        compactor_param.grad.data = mask * compactor_param.grad.data
        lasso_grad = compactor_param.data * ((compactor_param.data ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5))
        compactor_param.grad.data.add_(lasso_grad * opt.lasso_strength)
    return compactor_mask_dict


def Resrep_1(opt, iteration, model, ResRep_object, epoch, num_images, logger=None, device=None, flag=None):
    if hasattr(model, 'module'):
        flag = 1
    dummy_input = torch.randn([1, 3, opt.img_size[0], opt.img_size[0]]).to(device, non_blocking=True)

    iteration += 1
    interval = opt.print_freq / (torch.cuda.device_count() * opt.batch_size / 32) // (torch.cuda.device_count() * num_images / 50000)

    if epoch > 5 * torch.cuda.device_count() and iteration % interval == 0:
        layer_ones_dict, metric_dict = get_compactor_mask_score(model, ResRep_object)
        sorted_metric_dict = sorted(metric_dict, key=metric_dict.get)
        cur_flops = calculate_model_flops(model, dummy_input, ResRep_object, flag)
        cur_deactivated = compute_deactivated(model, ResRep_object.original_layer_ones_dict, layer_ones_dict)
        
        if cur_flops > opt.flops_target * ResRep_object.original_flops:
            next_deactivated_max = cur_deactivated + opt.begin_granularity
        else:
            next_deactivated_max = float('inf')
        if iteration % (10 * interval) == 0:
            logger.info(f'cur flops: {cur_flops}\t cur deactivated: {cur_deactivated}\t')
            for k, masked_channel in enumerate(sorted_metric_dict[:5]):
                logger.info(f'masked channel: {masked_channel[0]}_{masked_channel[1]:<6}  scores: {metric_dict[masked_channel]}')
            for k, masked_channel in enumerate(sorted_metric_dict[(cur_deactivated - 5):(cur_deactivated + 5)]):
                logger.info(f'masked channel: {masked_channel[0]}_{masked_channel[1]:<6}  scores: {metric_dict[masked_channel]}')
            logger.info(f'{epoch}: =======================')

        increase = order_idx = 0
        skip_id, indices = [], []
        while True:
            if cur_flops <= opt.flops_target * ResRep_object.original_flops:
                break
            layer, channel_id = sorted_metric_dict[order_idx]
            indices.append(order_idx)
            order_idx += 1
            if visit_module(model, ResRep_object.target_module_dict[layer.replace('module.', '')], flag).get_num_mask_ones() <= opt.num_at_least:
                skip_id.append(increase)
                increase += 1
                continue
            increase += 1
            if increase >= next_deactivated_max:
                break

        layer_masked_out_filters = defaultdict(list)
        for k in range(increase):
            if k not in skip_id:
                layer_masked_out_filters[sorted_metric_dict[indices[k]][0]].append(
                    sorted_metric_dict[indices[k]][1])

        set_model_masks(model, layer_masked_out_filters, ResRep_object)
        compactor_mask_dict = get_compactor_mask_dict(model)
        return model, iteration, compactor_mask_dict
    else:
        compactor_mask_dict = get_compactor_mask_dict(model)
        return model, iteration, compactor_mask_dict
