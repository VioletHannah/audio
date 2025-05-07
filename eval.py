#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-24 12:44
# @Author : 箴澄
# @Site : 评估model文件好不好，在测试集上运行代码，看看效果
# @File : eval.py
# @Software: PyCharm
from load_data import AudioDoADataset
from train import heatmapLoss
from MultiSource_3DCNN_mapNet import MultiSource3DCNNMapNet
from logger import *

from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

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


def transfer_to_vector(azimuth, elevation):
    """
    将方位角和俯仰角转换为三维向量
    参数：
    - azimuth: 方位角（弧度）
    - elevation: 俯仰角（弧度）
    返回：
    - 三维向量
    """
    x = math.cos(azimuth) * math.cos(elevation)
    y = math.sin(azimuth) * math.cos(elevation)
    z = math.sin(elevation)
    return np.array([x, y, z])


def calculate_spatial_angle(azimuth_label, elevation_label, azimuth_true, elevation_true):
    # 将标签和真实方向转换为三维向量
    label_vector = transfer_to_vector(azimuth_label, elevation_label)
    true_vector = transfer_to_vector(azimuth_true, elevation_true)

    # 计算点积
    dot_product = label_vector @ true_vector

    # 确保点积在有效范围内
    dot_product = max(min(dot_product, 1.0), -1.0)

    # 计算空间角（弧度）
    spatial_angle = math.acos(dot_product)

    return spatial_angle


def evaluate_model(dataset_path, model_path='sound_model.pth'):
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiSource3DCNNMapNet().to(device)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    # 2. 加载测试数据集
    test_dataset = AudioDoADataset(root_dir=dataset_path, split="test", n_channels=64)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. 初始化存储
    # true_azimuth = []
    # true_elevation = []
    # pred_azimuth = []
    # pred_elevation = []

    with (torch.no_grad()):
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = heatmapLoss(pred, y)
            logger.info(f"Test Loss: {loss.item():.6f}")

            row_num = 4
            col_num = 4
            row_pred = []
            row_gt = []
            for h in range(row_num):
                col_pred = []
                col_gt = []
                for w in range(col_num):
                 pred_slice = pred[h * col_num + w].cpu().detach().numpy()
                 heatmap_slice = y[h * col_num + w].cpu().detach().numpy()
                 pred_slice = np.pad(pred_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
                 heatmap_slice = np.pad(heatmap_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
                 col_pred.append(pred_slice)
                 col_gt.append(heatmap_slice)
                row_pred.append(np.concatenate(col_pred, axis=1))
                row_gt.append(np.concatenate(col_gt, axis=1))
            result = np.concatenate(row_pred, axis=0)
            gt = np.concatenate(row_gt, axis=0)
            result_uint8 = (result * 255).astype('uint8')
            cv2.imwrite("evalresult.png", result_uint8)
            gt_uint8 = (gt * 255).astype('uint8')
            cv2.imwrite("evalgt.png", gt_uint8)

            # true_azimuth.extend(y[:, 0].cpu().numpy().tolist())
            # true_elevation.extend(y[:, 1].cpu().numpy().tolist())

            # 预测和计算误差
            # pred_azimuth.extend(pred[:, 0].cpu().numpy().tolist())
            # pred_elevation.extend(pred[:, 1].cpu().numpy().tolist())


    # plot_joint_error_heatmap(true_azimuth, true_elevation, pred_azimuth, pred_elevation)


"""

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
"""
    # 5. 可视化误差分布


    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(121)
    # plot_polar_heatmap(all_azimuth_errors, all_elevation_errors,
    #                    "方位角-俯仰角联合误差分布")
    #
    # plt.subplot(122)
    # plt.hist2d(all_azimuth_errors, all_elevation_errors,
    #            bins=(30, 20), cmap='viridis')
    # plt.colorbar(label='样本数量')
    # plt.xlabel('方位角误差 (°)')
    # plt.ylabel('俯仰角误差 (°)')
    # plt.title("二维直方图误差分布")
    #
    # plt.tight_layout()
    # plt.savefig('error_analysis.png')
    # plt.show()

if __name__ == "__main__":
    evaluate_model(dataset_path="/home/zengkehan/voice/multisource_dataset",
                   model_path="/home/zengkehan/ssl/mulsource_sound_model91.pth")
