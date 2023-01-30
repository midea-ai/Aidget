# Copyright (c) Midea Group
# Licensed under the Apache License 2.0

from .loss_zoo import DistillKL, VIDLoss, IRGLoss, YOLOv7KDLoss

loss_dict = {'vanilla_kd': DistillKL, 'vid': VIDLoss, 'irg': IRGLoss, 'YoloKDLoss': YOLOv7KDLoss}
