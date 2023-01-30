# Copyright (c) Midea Group
# Licensed under the Apache License 2.0
import torch


def _setattr(model, name, module):
    name_list = name.split(".")
    part_name = name_list[:-1]
    cur_model = model
    for pn in part_name:
        cur_model = getattr(cur_model, pn)
    setattr(cur_model, name_list[-1], module)


class LayerInfo:
    def __init__(self, name, module: torch.nn.Module):
        self.module = module
        self.name = name
        self.type = type(module).__name__


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type):
        super().__init__()
        self.module = module
        self.name = module_name
        self.type = module_type
        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))

    def forward(self, *inputs):
        self.module.weight.data = self.module.weight.data.mul_(self.weight_mask)
        return self.module(*inputs)
