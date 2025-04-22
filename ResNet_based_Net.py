#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/16 15:07
# @Author : 箴澄
# @Site : 
# @File : ResNet_based_Net.py
# @Software: PyCharm

import torch
# import torchaudio
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class MicArrayResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MicArrayResNet, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        # 加载预训练的ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # 修改第一层接受单通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        # 使用ResNet的其他组件
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 定义回归头
        self.fc = nn.Linear(512, 128)
        self.doa = nn.Linear(128, 2)
        # self.fc_azimuth = nn.Linear(128, 1)
        # self.fc_elevation = nn.Linear(128, 1)

        # 初始化最后的全连接层
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.doa.weight)
        # nn.init.kaiming_normal_(self.fc_azimuth.weight)
        # nn.init.kaiming_normal_(self.fc_elevation.weight)

    def forward(self, x):
        x = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))

        # 使用tanh提供更强的梯度
        # azimuth = torch.tanh(self.fc_azimuth(x)) * torch.pi + torch.pi
        # elevation = torch.tanh(self.fc_elevation(x)) * 0.25 * torch.pi + 0.25 * torch.pi
        doa = self.doa(x)
        azimuth = torch.tanh(doa[:, 0]) * torch.pi + torch.pi # 0~2pi
        elevation = torch.tanh(doa[:, 1]) * 0.25 * torch.pi + 0.25 * torch.pi # 0~pi/2

        return torch.stack([azimuth.squeeze(), elevation.squeeze()], dim=1)


    # def preprocess_mic_array_data(audio_data, using_pretrained):
    #     """
    #     将麦克风阵列数据预处理为适合CNN输入的格式
    #     """
    #     # 降采样到8000 Hz
    #     resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    #     audio_data = resampler(audio_data)
    #
    #     # 为预训练网络调整尺寸
    #     # ResNet期望输入为224x224，我们可以进行调整
    #     if using_pretrained:
    #         # 将(1, 8000, 64)调整到类似图像的尺寸
    #         # 可以使用插值方法或裁剪加填充的方式
    #         audio_data = F.interpolate(audio_data, size=(224, 224), mode='bilinear')
    #
    #     # 归一化
    #     audio_data = (audio_data - audio_data.mean()) / (audio_data.std() + 1e-8)
    #
    #     return audio_data


# x = torch.randn(2, 1, 8000, 64)  # 假设输入为(1, 8000, 64)
# model = MicArrayResNet(pretrained=True)
# output = model(x)
# print(output)  # 输出形状应该为(1, 2)，表示DoA
