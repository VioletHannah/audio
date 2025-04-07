#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-12 22:04
# @Author : 箴澄
# @File : max_corr_backbone.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from BandPassFilterBank import MaxCorr
import numpy as np
import math

from numpy.ma import reshape


# class MaxCorr(nn.Module):
#     """
#     基于带有时移的带通滤波器 SincNet 的 MaxCorr 层
#     g(n, f1, f2, t1, ..., tc) = [k[n+t1], ..., k[n+tc]]
#     其中，f1、f2 为滤波器参数，t1...tc 为可学习的时移参数
#     """
#
#     def __init__(self, in_channels=1, out_channels=256, kernel_size=251, stride=75,
#                  sample_rate=64000, min_freq=50, num_shifts=16):
#         super(MaxCorr, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.sample_rate = sample_rate
#         self.min_freq = min_freq
#         self.num_shifts = num_shifts  # c in the equation
#
#         # 初始化滤波器库 (f1, f2 parameters)
#         self.freq_low = nn.Parameter(torch.zeros(out_channels))
#         self.freq_high = nn.Parameter(torch.zeros(out_channels))
#
#         # 初始化时移参数 (t1...tc)
#         self.time_shifts = nn.Parameter(torch.zeros(out_channels, num_shifts))
#
#         # 使用 MEL 初始化滤波器
#         self.init_filters()
#
#     def init_filters(self):
#         # 使用梅尔标度初始化滤波器库
#         low_freq = 50  # Min frequency in Hz
#         high_freq = self.sample_rate / 2 - (self.sample_rate / self.kernel_size)  # Max frequency
#
#         mel_low = 2595 * np.log10(1 + low_freq / 700)
#         mel_high = 2595 * np.log10(1 + high_freq / 700)
#
#         mel_points = torch.linspace(mel_low, mel_high, self.out_channels + 1)
#         f_points = 700 * (10 ** (mel_points / 2595) - 1)
#
#         # 使用随机偏移进行初始化
#         self.freq_low.data = f_points[:self.out_channels]
#         self.freq_high.data = f_points[1:]
#
#         # 在 0 和 kernel_size / 2 之间初始化时移
#         self.time_shifts.data = torch.rand(self.out_channels, self.num_shifts) * (self.kernel_size // 2)
#
#     def sinc(self, x):
#         # Sinc function: sin(x) / x
#         x = x + 1e-12  # Avoid division by zero
#         return torch.sin(x) / x
#
#     def create_filters(self):
#         # 将频率转换为标准化频率
#         low = self.freq_low / (self.sample_rate / 2)
#         high = self.freq_high / (self.sample_rate / 2)
#
#         band = (high - low)[:, None]
#         f_times_t = torch.linspace(0, 1, steps=self.kernel_size)[None, :]
#         f_times_t = f_times_t * self.sample_rate
#
#         # 计算每个频段的滤波器
#         # Low pass filter with cutoff frequency = high
#         lp_filter = 2 * high[:, None] * self.sinc(2 * math.pi * high[:, None] * f_times_t)
#         # High pass filter with cutoff frequency = low
#         hp_filter = 2 * low[:, None] * self.sinc(2 * math.pi * low[:, None] * f_times_t)
#         # Band pass filter = low pass - high pass
#         band_pass = lp_filter - hp_filter
#
#         # Apply Hamming window
#         window = torch.hamming_window(self.kernel_size, dtype=band_pass.dtype, device=band_pass.device)
#         band_pass = band_pass * window
#
#         # Normalize filter energy
#         band_pass = band_pass / (2 * band[:, None])
#
#         return band_pass
#
#     def apply_time_shifts(self, filters):
#         """Apply time shifts to filters based on learnable parameters"""
#         batch_size, channels, length = filters.shape
#         shifted_filters = []
#
#         for i in range(self.num_shifts):
#             shifts = self.time_shifts[:, i].long()
#             # Ensure shifts are within bounds
#             shifts = torch.clamp(shifts, 0, self.kernel_size - 1)
#
#             # Apply different shift to each filter
#             shifted = torch.zeros_like(filters)
#             for j in range(channels):
#                 shift = shifts[j]
#                 if shift > 0:
#                     shifted[0, j, shift:] = filters[0, j, :-shift]
#                 else:
#                     shifted[0, j, :] = filters[0, j, :]
#
#             shifted_filters.append(shifted)
#
#         # Stack shifted filters along new dimension
#         return torch.stack(shifted_filters, dim=1)
#
#     def forward(self, x):
#         filters = self.create_filters()
#
#         # Reshape filters for convolution [out_channels, in_channels, kernel_size]
#         filters = filters.view(self.out_channels, 1, self.kernel_size)
#
#         # Apply time shifts to filters (creating c shifted versions)
#         filters_expanded = filters.expand(1, -1, -1, -1)  # Add batch dimension
#         shifted_filters = self.apply_time_shifts(filters_expanded)
#
#         # Apply each shifted filter and get c outputs
#         outputs = []
#         for i in range(self.num_shifts):
#             current_filters = shifted_filters[:, i, :, :]
#             out = F.conv1d(x, current_filters, stride=self.stride, padding=(self.kernel_size - 1) // 2, groups=1)
#             outputs.append(out)
#
#         # Combine outputs (taking max as implied by the MaxCorr name)
#         outputs = torch.stack(outputs, dim=2)  # Shape: [B, out_channels, num_shifts, length]
#         outputs, _ = torch.max(outputs, dim=2)  # Max across shifts
#
#         return outputs

class MaxCorr(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, sr=48000):
        super().__init__()
        self.num_filters = num_filters # 滤波器数量
        self.kernel_size = kernel_size # 核大小，指截断的sinc函数长度
        self.in_channels = in_channels # 输入通道数
        self.sr = sr # 采样率sample rate

        # 初始化可学习参数：每个滤波器的f1, f2和64个时间延迟t_i
        self.f1 = nn.Parameter(torch.Tensor(num_filters))
        self.f2 = nn.Parameter(torch.Tensor(num_filters))
        self.ti = nn.Parameter(torch.Tensor(num_filters, in_channels))


        # 参数初始化
        nn.init.uniform_(self.f1, 0.0, sr/2)  # 初始频率范围0~Nyquist
        nn.init.uniform_(self.f2, 0.0, sr/2)
        nn.init.uniform_(self.ti, -kernel_size//2, kernel_size//2)  # 时间延迟范围

    def _sinc_kernel(self, f1, f2):
        """生成基础sinc带通滤波核"""
        n = torch.arange(self.kernel_size).to(f1.device) - (self.kernel_size - 1) // 2
        n = n.view(1, -1)  # [1, kernel_size]

        w1 = 2 * torch.pi * f1.view(-1, 1) / self.sr  # [num_filters, 1]
        w2 = 2 * torch.pi * f2.view(-1, 1) / self.sr  # [num_filters, 1]

        term1 = 2 * f2.view(-1, 1) * torch.sinc(w2 * n)  # [num_filters, kernel_size]
        term2 = 2 * f1.view(-1, 1) * torch.sinc(w1 * n)  # [num_filters, kernel_size]
        kernel = term1 - term2  # [num_filters, kernel_size]
        return kernel

    def _shift_kernel(self, kernel, ti):
        """应用时间延迟并插值（双线性插值）"""
        # kernel: [num_filters, in_channels, kernel_size]
        # ti: [num_filters, in_channels]

        # 生成基础网格（x轴：时间位移，y轴：通道索引）
        # ------------------------------------------------------------------
        # 1. 生成 x 轴坐标（时间维度）
        x_grid = torch.arange(self.kernel_size).to(kernel.device) - (self.kernel_size - 1) // 2
        x_grid = x_grid.view(1, 1, -1) - ti.unsqueeze(-1)  # [num_filters, in_channels, kernel_size]

        # 2. 归一化 x 轴到 [-1, 1]
        max_shift_x = (self.kernel_size // 2) * 1.0
        x_grid = x_grid / max_shift_x  # [num_filters, in_channels, kernel_size]

        # 3. 生成 y 轴坐标（通道维度，从 -1 到 1）
        y_grid = torch.linspace(-1, 1, kernel.size(1)).to(kernel.device)  # [in_channels]
        y_grid = y_grid.view(1, -1, 1, 1)  # [1, in_channels, 1, 1]

        # 4. 广播 x 和 y 网格并拼接
        x_grid = x_grid.unsqueeze(-1)  # [num_filters, in_channels, kernel_size, 1]
        y_grid = y_grid.expand(x_grid.size(0), -1, x_grid.size(2), -1)  # [num_filters, in_channels, kernel_size, 1]
        grid = torch.cat([x_grid, y_grid], dim=-1)  # [num_filters, in_channels, kernel_size, 2]

        # 调整卷积核形状以适应 grid_sample
        # ------------------------------------------------------------------
        # 输入形状需为 [N, C, H, W] = [num_filters, 1, in_channels, kernel_size]
        kernel = kernel.unsqueeze(1)  # 添加通道维度

        # 执行双线性插值
        shifted_kernel = F.grid_sample(
            kernel,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # 输出形状 [num_filters, 1, in_channels, kernel_size]

        # 恢复输出形状
        return shifted_kernel.squeeze(1)  # [num_filters, in_channels, kernel_size]

    def forward(self, x):
        B, C, T = x.shape
        assert C == self.in_channels
        # x = x.unsqueeze(1)

        # 1. 生成基础sinc核
        f1 = torch.sigmoid(self.f1) * (self.sr / 2)  # 限制频率范围
        f2 = torch.sigmoid(self.f2) * (self.sr / 2)
        base_kernel = self._sinc_kernel(f1, f2)  # [num_filters, kernel_size]

        # 2. 扩展为多通道并应用时间延迟
        base_kernel = base_kernel.unsqueeze(1).repeat(1, self.in_channels, 1)  # [num_filters, in_channels, kernel_size]
        shifted_kernels = self._shift_kernel(base_kernel, self.ti)  # [num_filters, in_channels, kernel_size]

        # 3. 对每个滤波器执行多通道卷积并求和
        reshaped_kernels = shifted_kernels.view(self.num_filters, self.in_channels, 1, self.kernel_size)  # [num_filters, in_channels, 1, kernel_size]
        x_reshaped = x.unsqueeze(2)  # [B, in_channels, 1, T]
        conv_out = F.conv2d(
            x_reshaped,
            reshaped_kernels,
            padding=(0, (self.kernel_size - 1) // 2),
            groups=1
        )
        output = conv_out.squeeze(2)
        return output

        # output = []
        # for i in range(self.num_filters):
        #     kernel = shifted_kernels[i]  # [in_channels, kernel_size]
        #     conv = F.conv1d(
        #         x,
        #         kernel.view(self.in_channels, 1, self.kernel_size),
        #         padding=(self.kernel_size - 1) // 2,
        #         groups=self.in_channels
        #     )  # [B, in_channels, T]
        #     output.append(conv.sum(dim=1, keepdim=True))  # [B, 1, T]
        # return torch.cat(output, dim=1)  # [B, num_filters, T]


class SoundDetBackbone(nn.Module):
    def __init__(self, input_shape=(1, 200, 256)):
        super(SoundDetBackbone, self).__init__()
        # MaxCorr layer
        self.maxcorr = MaxCorr(in_channels=64, num_filters=256, kernel_size=251, sr=48000)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=2),
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

        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512)
        )

        # Bidirectional GRU
        self.bigru = nn.GRU(512, 512, bidirectional=True, batch_first=True)

        # Additional layers for event classification
        self.conv_add = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # GAP
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer for regression
        self.fc = nn.Linear(512, 2)


    def forward(self, x):
        # Apply MaxCorr layer (SincNet with time shifts)
        # Input shape: [B, 1, T]
        x = self.maxcorr(x)
        # [B, 256, T]
        # print("mc", x.shape)

        x = self.conv_layers(x)
        # [B, C=1024, T/128]
        # print("conv", x.shape)
        # x = F.relu(self.bn1(self.conv1(x)))  # [B, 128, T/10]
        # x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, T/20]
        # x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, T/40]
        # x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, T/80]
        # x = F.relu(self.bn5(self.conv5(x)))  # [B, 1024, T/160]

        # Apply deconvolutional layers
        x = self.deconv_layers(x)
        # [B, C=512, T/32]
        # print("deconv", x.shape)
        # x = F.relu(self.bn_deconv1(self.deconv1(x)))  # [B, 512, T/80]
        # x = F.relu(self.bn_deconv2(self.deconv2(x)))  # [B, 512, T/40]

        # Apply Bidirectional GRU
        x = x.transpose(1, 2)
        # [B, T/32, C=512]
        gru_output, _ = self.bigru(x)
        # [B, T/32, C=512]
        x = gru_output[:, :, :512]

        x = x.transpose(1, 2)
        x = self.conv_add(x)  # [B, 512, T/40]
        # print("addconv", x.shape)

        x = self.avgpool(x)
        # print("avg", x.shape)
        features = x.transpose(1, 2).squeeze()  # [B, 512]

        # Apply fully connected layer for regression
        DoA = self.fc(features) # [B, 2]
        x, y = DoA[:, 0], DoA[:, 1]
        x_limit = torch.sigmoid(x) * 2 * math.pi
        y_limit = torch.sigmoid(y) * (math.pi / 2)
        DoA = torch.stack([x_limit, y_limit], dim=1)

        return DoA


def test_sounddet_backbone():
    # Create a sample input: [batch_size, channels, time_length]
    batch_size = 4
    time_length = 48000
    x = torch.randn(batch_size, 64, time_length)
    x = x.cuda() if torch.cuda.is_available() else x

    # Create the model
    model = SoundDetBackbone()
    model = model.cuda() if torch.cuda.is_available() else model
    outputs = model(x)

    print(outputs)

    # for key, value in outputs.items():
    #     print(f"{key} shape: {value.shape}")
        # print(value[0])
        # print(value[1])


    return model, outputs

# Example usage:
# model, outputs = test_sounddet_backbone()