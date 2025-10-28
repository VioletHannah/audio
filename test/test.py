#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-23 12:02
# @Author : 箴澄
# @Site : 
# @File : test.py
# @Software: PyCharm
import os
import random
import soundfile as sf
import torch
import torch.nn.functional as F
import pyroomacoustics as pra

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


import matplotlib.pyplot as plt
import pandas as pd

def data_correct_val():
    path = "/home/zengkehan/voice/speech_snr_30/wavs/room_0/mic_0.wav"
    data, sr = sf.read(path)
    df = pd.DataFrame(data)
    plt.plot(df)
    plt.show()
    plt.scatter(y=df[0], x=df.index)


# 创建自定义颜色映射：0.5以下为黑色，0.5-1为蓝到红
def create_custom_colormap(low_color='black'):
    """创建自定义颜色映射，将0.5-1映射到蓝红渐变"""
    colors = [(0, 0, 0), (0, 0, 1), (1, 0, 0)]  # 黑 -> 蓝 -> 红

    # 创建颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # 设置低于0.5的值显示为黑色
    cmap.set_under(low_color)
    return cmap


def visualize_heatmaps(pred_hm, heatmaps, row_num=4, col_num=4,
                       output_prefix='output', background='black'):
    """可视化热力图，将0.5-1的值映射到蓝红渐变"""
    # 创建自定义颜色映射
    cmap = create_custom_colormap(low_color=background)

    # 确保图像尺寸一致
    plt.rcParams['figure.figsize'] = (col_num * 4, row_num * 4)

    # 创建预测结果的可视化
    fig, axes = plt.subplots(row_num, col_num)
    axes = axes.flatten()

    for i, (ax, pred) in enumerate(zip(axes, pred_hm)):
        if i < len(pred_hm):
            # 将热力图数据转换为numpy数组
            pred_np = pred.cpu().detach().numpy() if hasattr(pred, 'cpu') else pred
            # 使用imshow显示热力图，设置颜色映射和范围
            im = ax.imshow(pred_np, cmap=cmap, vmin=0.5, vmax=1.0, interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Confidence')

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为颜色条腾出空间
    plt.savefig(f'{output_prefix}_pred.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 同样处理真实标签热力图
    fig, axes = plt.subplots(row_num, col_num)
    axes = axes.flatten()

    for i, (ax, gt) in enumerate(zip(axes, heatmaps)):
        if i < len(heatmaps):
            # 将热力图数据转换为numpy数组
            gt_np = gt.cpu().detach().numpy() if hasattr(gt, 'cpu') else gt

            # 使用imshow显示热力图，设置颜色映射和范围
            im = ax.imshow(gt_np, cmap=cmap, vmin=0.5, vmax=1.0, interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Confidence')

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为颜色条腾出空间
    plt.savefig(f'{output_prefix}_gt.png', dpi=300, bbox_inches='tight')
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



# blue_red_heatmap()

# 使用示例
def vis():
    pred_hm = np.zeros([128, 128])
    heatmaps = np.zeros([128, 128])
    # 在热力图中随机填充三个1
    for j in range(3):
        x = random.randint(0, 127)
        y = random.randint(0, 127)
        pred_hm[x, y] = 10
        heatmaps[x, y] = 10
    # 高斯扩散
    pred_hm = gaussian_filter(pred_hm, 2)
    heatmaps = gaussian_filter(heatmaps, 2)
    # 归一化到0-1
    pred_hm = pred_hm / np.max(pred_hm)
    heatmaps = heatmaps / np.max(heatmaps)

    # 可视化热力图，背景为白色，0-1映射到蓝红渐变
    color_map = LinearSegmentedColormap.from_list(
        'blue_red_gradient',  # 颜色映射名称
        [(0, 0, 1), (1, 0, 0)],  # RGB颜色值：蓝→红
        N=256  # 颜色分段数（建议≥256）
    )
    blue_red_heatmap(pred_hm, title="pred")
    blue_red_heatmap(heatmaps, title="gt")

def _shift_kernel(kernel, ti):
    """应用时间延迟并插值（双线性插值）"""
    # kernel: [num_filters, in_channels, kernel_size]
    # ti: [num_filters, in_channels]

    num_filters, in_channels, kernel_size = kernel.size()
    # 生成基础网格（x轴：时间位移，y轴：通道索引）
    # ------------------------------------------------------------------
    # 1. 生成 x 轴坐标（时间维度）
    x_grid = torch.arange(kernel_size).to(kernel.device) - (kernel_size-1)//2
    x_grid = x_grid.view(1, 1, -1) - ti.unsqueeze(-1)  # [num_filters, in_channels, kernel_size]

    # 2. 归一化 x 轴到 [-1, 1]
    max_shift_x = (kernel_size // 2) * 1.0
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
def try_shift_kernel():
    kernel = torch.Tensor([
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
        [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    ])  # shape [3, 2, 5]

    ti = torch.Tensor([
        [1.0, -1.0],   # 滤波器0: 通道0右移1位，通道1左移1位
        [0.0, 0.5],    # 滤波器1: 通道0不移动，通道1右移0.5位
        [-2.0, 2.0]    # 滤波器2: 通道0左移2位，通道1右移2位
    ])
    print(kernel.shape)
    print(kernel[0, 0])
    print(ti[0, 0])

    shifted_kernel = _shift_kernel(kernel, ti)
    print(shifted_kernel)

def generate_ssl_data():
    # 设置参数
    mic_num_per_line = 4
    mic_length = 1
    room_dimension = np.array([10, 10, 3])
    # num_samples = 1000

    # 线性麦克风阵列位置
    mic_positions = np.zeros((3, mic_num_per_line))
    mic_positions[1] = room_dimension[1]/2
    spacing = mic_length / (mic_num_per_line - 1)  # 麦克风间距
    offset = (mic_num_per_line - 1) / 2  # 中心对称偏移量
    for i in range(mic_num_per_line):
        mic_positions[0, i] = (i - offset)*spacing + room_dimension[0]/2

    # 声源位置
    source_pos = np.array([2, room_dimension[1]/2, 0])
    # 声源信号
    source_path = "/home/zengkehan/voice/google_speech_commands/go/"
    source_file = random.choice(os.listdir(source_path))
    audio, fs = sf.read(source_path + source_file)

    # 创建房间
    room = pra.ShoeBox(room_dimension, fs=fs, max_order=0, absorption=1.0)
    room.add_source(source_pos)
    room.sources[0].signal = audio
    room.add_microphone_array(mic_positions)
    room.plot()
    plt.show()

    # 生成数据
    room.simulate()
    # 检查音频数据
    audio_data = room.mic_array.signals
    print(f"音频数据形状: {audio_data.shape}")
    for i in range(audio_data.shape[0]):
        plt.plot(audio_data[i][1000:2000])
        plt.title(f"micphone {i + 1}")
        plt.show()
def try_ssl_data():
    generate_ssl_data()
