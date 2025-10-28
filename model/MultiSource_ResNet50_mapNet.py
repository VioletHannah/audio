#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/22 10:25
# @Author : 箴澄
# @Site : 
# @File : MultiSource_ResNet50_mapNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MicArrayResNet50HeatmapNetOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        # 载入预训练的 ResNet50
        resnet = models.resnet50(pretrained=True)

        # ----- Stage1: 处理第二空间维度 + 时间 -----
        self.stage1_conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.stage1_bn1 = nn.BatchNorm2d(64)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1_layer1 = resnet.layer1
        self.stage1_layer2 = resnet.layer2
        self.stage1_layer3 = resnet.layer3

        # ----- Stage2: 处理第一空间维度 + 时间 -----
        self.stage2_conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage2_bn1 = nn.BatchNorm2d(256)
        self.stage2_relu1 = nn.ReLU(inplace=True)
        self.stage2_layer2 = resnet.layer2
        # 只取前 3 层 Bottleneck
        self.stage2_layer3 = nn.Sequential(*list(resnet.layer3.children())[:3])

        # ----- 融合模块 Fusion Module -----
        # （不需要作为 nn.Module，直接在 forward 用 F.adaptive_avg_pool2d）
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # ----- Decoder -----
        self.decoder_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(512)
        self.decoder_relu1 = nn.ReLU(inplace=True)

        self.decoder_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_relu2 = nn.ReLU(inplace=True)

        # ----- UpSampling -----
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.upbn1 = nn.BatchNorm2d(128)
        self.uprelu1 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.upbn2 = nn.BatchNorm2d(64)
        self.uprelu2 = nn.ReLU(inplace=True)

        # ----- Heatmap Head -----
        self.heatmap_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.heatmap_conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x: [B, 1, 16000, 8, 8]
        # 先转置为 [B, 8, 8, 16000]
        x = x.permute(0, 3, 4, 2, 1).squeeze(-1)  # → [B, 8, 8, 16000]
        # x: [B, 8, 8, 16000]
        B = x.size(0)

        # ---- Stage1 ----
        x = self.stage1_conv1(x)  # → [B,64,4,8000]
        x = self.stage1_bn1(x);
        x = self.stage1_relu(x)
        x = self.stage1_maxp(x)  # → [B,64,2,4000]
        x = self.stage1_layer1(x)  # → [B,256,2,4000]
        x = self.stage1_layer2(x)  # → [B,512,1,2000]
        x = self.stage1_layer3(x)  # → [B,1024,1,1000]
        x1 = x.clone()  # 保存以供融合

        # ---- 转置 ----
        x = x.permute(0, 2, 1, 3)  # → [B,1,1024,1000]

        # ---- Stage2 ----
        x = self.stage2_conv1(x)  # → [B,256,1024,1000]
        x = self.stage2_bn1(x);
        x = self.stage2_relu1(x)
        x = self.stage2_layer2(x)  # → [B,512,512,500]
        x = self.stage2_layer3(x)  # → [B,1024,256,250]
        x2 = x

        # ---- Fusion Module ----
        # 将 x1 的时间维下采样到 250，再 expand 到 [B,1024,256,250]
        x1_pool = F.adaptive_avg_pool2d(x1, output_size=(1, x2.size(-1)))  # [B,1024,1,250]
        x1_exp = x1_pool.expand(-1, -1, x2.size(2), -1)  # [B,1024,256,250]
        x_fuse = torch.cat([x2, x1_exp], dim=1)  # [B,2048,256,250]
        x = self.fuse_conv(x_fuse)  # [B,1024,256,250]

        # ---- Decoder ----
        x = self.decoder_conv1(x)  # → [B,512,256,250]
        x = self.decoder_bn1(x);
        x = self.decoder_relu1(x)
        x = self.decoder_conv2(x)  # → [B,256,256,250]
        x = self.decoder_bn2(x);
        x = self.decoder_relu2(x)

        # ---- UpSampling ----
        x = self.upconv1(x)  # → [B,128,512,500]
        x = self.upbn1(x);
        x = self.uprelu1(x)
        x = self.upconv2(x)  # → [B,64,1024,1000]
        x = self.upbn2(x);
        x = self.uprelu2(x)

        # ---- Heatmap Head ----
        x = self.heatmap_conv1(x)  # → [B,32,1024,1000]
        x = self.heatmap_conv2(x)  # → [B, 1,1024,1000]

        # ---- 最终下采样 + 概率化 ----
        x = F.adaptive_avg_pool2d(x, output_size=(128, 128))  # → [B,1,91,360]
        x = x.squeeze(1)  # → [B,91,360]
        # x = F.softmax(x.view(B, -1), dim=1).view(B, 91, 360)
        x = torch.sigmoid(x)

        return x


