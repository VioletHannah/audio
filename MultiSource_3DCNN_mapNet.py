#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/22 14:08
# @Author : 箴澄
# @Site : 
# @File : MultiSource_3DCNN_mapNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiSource3DCNNMapNet(nn.Module):
    def __init__(self):
        super(MultiSource3DCNNMapNet, self).__init__()
        # [B, 1, 16000, 8, 8]
        # 3D卷积层
        self.Conv3Dstack1 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2, 4, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 8, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.Conv3Dstack2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.Conv3Dstack3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.Conv3D = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1)

        # 2D卷积层
        self.conv2d = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        # 反卷积层
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32→64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 64→128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 128→256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, 16000, 8, 8]
        x = self.Conv3Dstack1(x)
        # x: [B, 8, 800, 8, 8]
        x = self.Conv3Dstack2(x)
        # x: [B, 64, 40, 8, 8]
        x = self.Conv3Dstack3(x)
        # x: [B, 512, 2, 8, 8]
        x = self.Conv3D(x).squeeze()
        # x: [B, 1024, 8, 8]

        x = self.conv2d(x)
        # x: [B, 512, 8, 8]

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x: [B, 32, 128, 128]
        x = self.final(x).squeeze()
        # x: [B, 128, 128]
        x = torch.sigmoid(x)
        # x: [B, 128, 128]

        return x


# m = MultiSource3DCNNMapNet()
# x = torch.randn(2, 1, 16000, 8, 8)
# y = m(x)
# print(y.shape)