#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/10 11:19
# @Author : 箴澄
# @Site : 
# @File : synperiodic_backbone.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from SynperiodicFilterBanks import FilterBank


class SoundSynpBackbone(nn.Module):
    def __init__(self, input_shape=(64, 16000), filter_group=0, sample_rate=16000):
        """
        初始化网络，使用SynperiodicFilterBanks替代MaxCorr

        Args:
            input_shape: 输入音频形状 (channels, time_length)
            filter_group: 使用的滤波器组索引 (0, 1, 2 分别对应group1, group2, group3)
            sample_rate: 采样率，默认16000Hz
        """
        super(SoundSynpBackbone, self).__init__()

        self.in_channels = input_shape[0]  # 64通道
        self.filter_group = filter_group  # 滤波器组索引
        self.filter_num = 256  # 滤波器数量
        self.kernel_length = 1025  # 滤波器核长度，保持奇数

        # 初始化SynperiodicFilterBanks
        self.filter_bank = FilterBank(
            n_fft=2048,
            kernel_length=self.kernel_length,
            filter_num=self.filter_num,
            sample_rate=sample_rate,
            min_freq=10,
            window='hann',
            filter_type='melscale',  # 使用梅尔尺度初始化
            filterbank_type='synperiodic'
        )

        # 卷积层网络结构保持不变
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.filter_num, 128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 1024, kernel_size=3, stride=2),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 反卷积层
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512)
        )

        # 双向GRU
        self.bigru = nn.GRU(512, 512, bidirectional=True, batch_first=True)

        # 附加卷积层
        self.conv_add = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层用于回归
        self.fc = nn.Linear(512, 2)


    def apply_synperiodic_filterbank(self, x, filters):
        """
        应用同步周期性滤波器组到输入音频

        Args:
            x: 输入音频 [batch_size, channels, time_length]
            filters: 滤波器组 [filter_num, kernel_length]

        Returns:
            滤波后的特征表示 [batch_size, filter_num, time_length]
        """
        batch_size, channels, time_length = x.shape
        output = torch.zeros(batch_size, self.filter_num, time_length, device=x.device)

        # 确保滤波器是实数
        filters = filters.real

        # 对每个通道分别应用滤波器
        for c in range(channels):
            channel_input = x[:, c:c + 1, :]  # [batch_size, 1, time_length]

            # 将滤波器重塑为卷积核形状
            conv_filters = filters.view(self.filter_num, 1, -1)  # [filter_num, 1, kernel_length]

            # 应用卷积
            channel_output = F.conv1d(
                channel_input,
                conv_filters,
                padding=(self.kernel_length - 1) // 2
            )  # [batch_size, filter_num, time_length]

            # 累加每个通道的输出
            output += channel_output

        # 返回所有通道的平均结果
        return output / channels

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入音频 [batch_size, channels, time_length]

        Returns:
            DoA: 到达角(Direction of Arrival) [batch_size, 2]
        """
        # 获取滤波器组
        filter_group1_list, filter_group2_list, filter_group3_list = self.filter_bank.obtain_filter_bank()

        # 根据指定的滤波器组索引选择滤波器
        if self.filter_group == 0:
            # 使用第一组滤波器
            filters = filter_group1_list[0]
        elif self.filter_group == 1:
            # 使用第二组滤波器
            filters = filter_group2_list[0]
        else:
            # 使用第三组滤波器
            filters = filter_group3_list[0]

        # 应用同步周期性滤波器组
        x = self.apply_synperiodic_filterbank(x, filters)

        # 应用卷积层
        x = self.conv_layers(x)

        # 应用反卷积层
        x = self.deconv_layers(x)

        # 应用双向GRU
        x = x.transpose(1, 2)  # [batch_size, time_steps, features]
        gru_output, _ = self.bigru(x)
        x = gru_output[:, :, :512]  # 取前向GRU的输出

        # 转回卷积形式
        x = x.transpose(1, 2)  # [batch_size, features, time_steps]
        x = self.conv_add(x)

        # 全局平均池化
        x = self.avgpool(x)
        features = x.transpose(1, 2).squeeze()  # [batch_size, 512]

        # 应用全连接层得到DoA预测
        DoA = self.fc(features)  # [batch_size, 2]

        # 限制输出范围
        x, y = DoA[:, 0], DoA[:, 1]
        x_limit = torch.sigmoid(x) * 2 * math.pi
        y_limit = torch.sigmoid(y) * (math.pi / 2)
        DoA = torch.stack([x_limit, y_limit], dim=1)

        return DoA


def test_sounddet_backbone():
    """
    测试SoundDetBackbone模型
    """
    # 创建样本输入: [batch_size, channels, time_length]
    batch_size = 4
    time_length = 16000  # 1秒的16kHz音频
    channels = 64
    x = torch.randn(batch_size, channels, time_length)

    # 如果有GPU则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    # 测试每个滤波器组
    results = []
    for filter_group in range(3):
        print(f"Testing with filter group {filter_group + 1}...")
        model = SoundSynpBackbone(
            input_shape=(channels, time_length),
            filter_group=filter_group,
            sample_rate=16000
        ).to(device)

        # 前向传播
        outputs = model(x)
        print(f"Output shape: {outputs.shape}")
        print(f"Output values: {outputs[0]}")  # 打印第一个样本的输出
        results.append(outputs)

    return results


# 带有频率子带切分的增强版本
class EnhancedSoundDetBackbone(nn.Module):
    def __init__(self, input_shape=(64, 16000), sample_rate=16000):
        """
        使用多个频率子带的增强版SoundDetBackbone

        Args:
            input_shape: 输入音频形状 (channels, time_length)
            sample_rate: 采样率，默认16000Hz
        """
        super(EnhancedSoundDetBackbone, self).__init__()

        self.in_channels = input_shape[0]
        self.filter_num = 256
        self.kernel_length = 501

        # 初始化SynperiodicFilterBanks
        self.filter_bank = FilterBank(
            n_fft=2048,
            kernel_length=self.kernel_length,
            filter_num=self.filter_num,
            sample_rate=sample_rate,
            min_freq=10,
            window='hann',
            filter_type='melscale',
            filterbank_type='synperiodic'
        )

        # 使用更多通道的卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.filter_num * 3, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # 其余层与之前相同
            nn.Conv1d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 1024, kernel_size=3, stride=2),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 保持其他层不变
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512)
        )

        self.bigru = nn.GRU(512, 512, bidirectional=True, batch_first=True)

        self.conv_add = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 2)

    def apply_synperiodic_filterbank(self, x, filters):
        """与基础版本相同的滤波器应用函数"""
        batch_size, channels, time_length = x.shape
        output = torch.zeros(batch_size, self.filter_num, time_length, device=x.device)

        filters = filters.real

        for c in range(channels):
            channel_input = x[:, c:c + 1, :]
            conv_filters = filters.view(self.filter_num, 1, -1)

            channel_output = F.conv1d(
                channel_input,
                conv_filters,
                padding=(self.kernel_length - 1) // 2
            )

            output += channel_output

        return output / channels

    def forward(self, x):
        # 获取所有滤波器组
        filter_group1_list, filter_group2_list, filter_group3_list = self.filter_bank.obtain_filter_bank()

        # 应用所有三组滤波器
        x1 = self.apply_synperiodic_filterbank(x, filter_group1_list[0])
        x2 = self.apply_synperiodic_filterbank(x, filter_group2_list[0])
        x3 = self.apply_synperiodic_filterbank(x, filter_group3_list[0])

        # 连接所有滤波器组的输出
        x = torch.cat([x1, x2, x3], dim=1)  # [batch_size, filter_num*3, time_length]

        # 应用其余网络层
        x = self.conv_layers(x)
        x = self.deconv_layers(x)

        x = x.transpose(1, 2)
        gru_output, _ = self.bigru(x)
        x = gru_output[:, :, :512]

        x = x.transpose(1, 2)
        x = self.conv_add(x)

        x = self.avgpool(x)
        features = x.transpose(1, 2).squeeze()

        DoA = self.fc(features)
        x, y = DoA[:, 0], DoA[:, 1]
        x_limit = torch.sigmoid(x) * 2 * math.pi
        y_limit = torch.sigmoid(y) * (math.pi / 2)
        DoA = torch.stack([x_limit, y_limit], dim=1)

        return DoA