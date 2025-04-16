#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/10 16:13
# @Author : 箴澄
# @Site : 
# @File : time_domain_cnn.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MicArrayLocalizationNet(nn.Module):
    def __init__(self):
        super(MicArrayLocalizationNet, self).__init__()

        # 降采样到8000Hz
        self.downsample = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        # 输入尺寸: (1, 8000, 64)

        # 第一层卷积：压缩时间维度
        # 输入: (1, 8000, 64), 输出: (16, 1000, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
        )

        # 第二层卷积：继续压缩时间维度
        # 输入: (16, 1000, 32), 输出: (24, 250, 16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
        )

        # 第三层卷积：再次压缩至目标时间维度
        # 输入: (24, 250, 16), 输出: (32, 125, 8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # 第四层卷积：仅压缩空间维度
        # 输入: (32, 125, 8), 输出: (32, 125, 4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        # 第五层卷积：特征提取，维度不变
        # 输入: (32, 125, 4), 输出: (32, 125, 4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 计算展平后的特征数量
        self.flatten_size = 32 * 125 * 4

        # 全连接层进行回归
        self.fc1 = nn.Linear(self.flatten_size, 256)
        nn.init.xavier_normal_(self.fc1.weight)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)

        # 最终输出层
        self.azimuth = nn.Linear(64, 1)
        self.elevation = nn.Linear(64, 1)

    def forward(self, x):
        x = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) + x

        x = x.view(-1, self.flatten_size)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        azimuth = torch.sigmoid(self.azimuth(x)) * 2 * torch.pi
        elevation = torch.sigmoid(self.elevation(x)) * 0.5 * torch.pi
        DoA = torch.stack([azimuth.squeeze(), elevation.squeeze()], dim=1)

        return DoA

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_MALNet_backbone():
    # Create a sample input: [batch_size, channels, time_length]
    batch_size = 2
    time_length = 8000
    x = torch.randn(batch_size, 1, time_length, 64)
    x = x.cuda() if torch.cuda.is_available() else x

    # Create the model
    model = MicArrayLocalizationNet()
    model = model.cuda() if torch.cuda.is_available() else model
    outputs = model(x)

    print(outputs)

# test_MALNet_backbone()