# Copyright (c) Midea Group
# Licensed under the Apache License 2.0
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 20, 3)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.max_pool2d(F.relu(out), (2, 2))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.max_pool2d(F.relu(out), 2)
        out = out.view(-1, int(out.nelement() / out.shape[0]))
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.fc2(out)
        return out
