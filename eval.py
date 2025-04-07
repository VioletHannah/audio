#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-24 12:44
# @Author : 箴澄
# @Site :
# @File : eval.py
# @Software: PyCharm
from load_data import AudioDoADataset
from max_corr_backbone import SoundDetBackbone
from train import AngleLoss

from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_polar_heatmap(azimuth_errors, elevation_errors, title=""):
    """
    极坐标热力图可视化函数
    参数：
    - azimuth_errors: 方位角误差列表
    - elevation_errors: 俯仰角误差列表
    """
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    hb = ax.hexbin(np.radians(azimuth_errors), elevation_errors,
                   gridsize=30, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='误差密度')
    plt.title(title)
    return ax


def calculate_spatial_angle(azimuth_label, elevation_label, azimuth_true, elevation_true):
    # 将标签方向转换为三维向量
    x1 = math.cos(azimuth_label) * math.cos(elevation_label)
    y1 = math.sin(azimuth_label) * math.cos(elevation_label)
    z1 = math.sin(elevation_label)

    # 将真实方向转换为三维向量
    x2 = math.cos(azimuth_true) * math.cos(elevation_true)
    y2 = math.sin(azimuth_true) * math.cos(elevation_true)
    z2 = math.sin(elevation_true)

    # 计算点积
    dot_product = x1 * x2 + y1 * y2 + z1 * z2

    # 确保点积在有效范围内
    dot_product = max(min(dot_product, 1.0), -1.0)

    # 计算空间角（弧度）
    spatial_angle = math.acos(dot_product)

    return spatial_angle


def evaluate_model(dataset_path, model_path='sound_model.pth'):
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoundDetBackbone().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 加载测试数据集
    test_dataset = AudioDoADataset(root_dir=dataset_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. 初始化存储
    all_azimuth_errors = []
    all_elevation_errors = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # 预测和计算误差
            pred = model(x)
            loss = AngleLoss(pred, y)
            total_loss += loss.item()

            # 转换为角度误差（假设输出为弧度）
            azimuth_rad_errors = torch.abs(pred[:, 0] - y[:, 0])
            elevation_rad_errors = torch.abs(pred[:, 1] - y[:, 1])

            # 处理方位角周期性（转换为度数）
            azimuth_deg_errors = torch.rad2deg(torch.min(azimuth_rad_errors,
                                                         2 * torch.pi - azimuth_rad_errors))
            elevation_deg_errors = torch.rad2deg(elevation_rad_errors)

            all_azimuth_errors.extend(azimuth_deg_errors.cpu().numpy())
            all_elevation_errors.extend(elevation_deg_errors.cpu().numpy())

    # 4. 打印统计信息
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"Azimuth MAE: {np.mean(all_azimuth_errors):.2f}° ± {np.std(all_azimuth_errors):.2f}°")
    print(f"Elevation MAE: {np.mean(all_elevation_errors):.2f}° ± {np.std(all_elevation_errors):.2f}°")

    # 5. 可视化误差分布
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plot_polar_heatmap(all_azimuth_errors, all_elevation_errors,
                       "方位角-俯仰角联合误差分布")

    plt.subplot(122)
    plt.hist2d(all_azimuth_errors, all_elevation_errors,
               bins=(30, 20), cmap='viridis')
    plt.colorbar(label='样本数量')
    plt.xlabel('方位角误差 (°)')
    plt.ylabel('俯仰角误差 (°)')
    plt.title("二维直方图误差分布")

    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.show()


if __name__ == "__main__":
    evaluate_model(dataset_path="/home/zengkehan/voice/single64_dataset",
                   model_path="sound_model.pth")