#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/25 00:42
# @Author : 箴澄
# @Func：一些函数
# @File : util.py
# @Software: PyCharm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import logging
import datetime

def get_logger(filename=None):
    # 如果未指定文件名，使用当前时间作为默认名
    if not filename:
        cnt_time = datetime.datetime.now()
        filename = f'./{cnt_time.strftime("%Y%m%d_%H%M%S")}.log'

    logger = logging.getLogger()
    # 避免重复添加处理器
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 文件处理器（使用传入的文件名）
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def azimuth_elevation_to_alpha_beta(azimuth, elevation):
    """
    将方位角和俯仰角转换为alpha和beta
    :param azimuth: 方位角（以度为单位）与 x 轴正方向的夹角（在 xy 平面上）
    :param elevation: 俯仰角（以度为单位）与 xy 平面的夹角（z 方向的仰角）
    :return: alpha和beta（z-x夹角和z-y夹角，以度为单位）
    """

    azi_rad = np.radians(azimuth)
    ele_rad = np.radians(elevation)
    tan_phi = np.tan(ele_rad)

    # 计算k
    k = np.where(tan_phi == 0, np.inf, 1 / tan_phi)  # 处理tan_phi=0的情况

    # 计算tan_alpha和tan_beta
    tan_alpha = np.cos(azi_rad) / np.tan(ele_rad)
    tan_beta = np.sin(azi_rad) / np.tan(ele_rad)

    # 计算alpha和beta
    alpha_rad = np.arctan(tan_alpha)
    beta_rad = np.arctan(tan_beta)

    # 转换为角度
    alpha = np.degrees(alpha_rad)
    beta = np.degrees(beta_rad)

    return alpha, beta

def alpha_beta_to_azimuth_elevation(alpha, beta):
    """
    将alpha和beta转换为方位角和俯仰角
    :param alpha: z-x夹角角度
    :param beta: z-y夹角角度
    :return: 方位角和俯仰角（以度为单位）
    """

    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    tan_alpha = np.tan(alpha_rad)
    tan_beta = np.tan(beta_rad)

    # 计算theta（方位角）
    theta_rad = np.arctan2(tan_beta, tan_alpha)  # 自动处理象限

    # 计算phi（俯仰角）
    k = np.sqrt(tan_alpha ** 2 + tan_beta ** 2)
    phi_rad = np.where(k == 0, np.pi / 2, np.arctan(1 / k))  # 处理k=0的情况

    # 转换为角度
    azimuth = np.degrees(theta_rad) % 360
    elevation = np.degrees(phi_rad)

    return azimuth, elevation

def search_source_position(heatmap, thresh=0.5):
    """
    在热图中搜索声源位置
    参数：
    - heatmap: 形状为[128, 128]的热图
    - thresh: 阈值
    返回：
    - 源位置张量，形状为[num, 2]
    """
    source_positions = []
    # 找到大于阈值的点
    while np.max(heatmap) > thresh:
        index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # 计算声源位置
        alpha = np.clip(index[0] - 63, -63, 64)
        beta = np.clip(index[1] - 63, -63, 64)
        azimuth, elevation = alpha_beta_to_azimuth_elevation(alpha, beta)
        source_positions.append([azimuth, elevation])
        # 将最大值及其周围置为0，避免重复搜索
        c = 3
        heatmap[index[0]-c:index[0]+c+1, index[1]-c:index[1]+c+1] = 0
    return source_positions

def blue_red_heatmap(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    if not isinstance(data, np.ndarray) or data.shape != (16, 128, 128):
        raise ValueError("输入 data 必须是形状 (16,128,128) 的 numpy 数组")

    # 修改1：定义新的颜色映射（蓝 → 白 → 红）
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝(-1) → 白(0) → 红(1)
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)

    # 修改2：调整标准化范围到 [-1, 1]
    norm = Normalize(vmin=-1, vmax=1)

    # 绘图
    fig, axs = plt.subplots(4, 4, figsize=(10, 8),
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.suptitle(title, fontsize=15, y=0.92)

    for idx, ax in enumerate(axs.flat):
        # 修改3：使用新定义的 cmap 和 norm
        im = ax.imshow(data[idx], cmap=cmap, norm=norm,
                       origin='lower', aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes((0.90, 0.12, 0.015, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs

def blue_red_heatmap_old(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    if not isinstance(data, np.ndarray) or data.shape != (16, 128, 128):
        raise ValueError("输入 data 必须是形状 (16,128,128) 的 numpy 数组")

    # 配色和标准化
    colors = [(1,1,1), (0.5,0.5,1), (1,0,0)]  # 白 → 蓝 → 红
    cmap = LinearSegmentedColormap.from_list("white_blue_red", colors, N=256)
    norm = Normalize(vmin=0, vmax=1)

    # 绘图
    fig, axs = plt.subplots(4,4, figsize=(10, 8),
                            gridspec_kw={'wspace':0.05,'hspace':0.05})
    fig.suptitle(title, fontsize=15, y=0.92)

    for idx, ax in enumerate(axs.flat):
        im = ax.imshow(data[idx], cmap='jet', norm=norm,
                       origin='lower', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes((0.90, 0.12, 0.015, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs

def blue_red_heatmap_new(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    """
    生成16张 128×128 的三点高斯热图，4×4 网格显示。
    支持负值到正值的范围，颜色映射：负值(白→蓝)，0值(蓝)，正值(蓝→红)
    """
    if not isinstance(data, np.ndarray) or data.shape != (16, 128, 128):
        raise ValueError("输入 data 必须是形状 (16,128,128) 的 numpy 数组")

    # 计算全局归一化范围（确保包含0）
    vmin = min(data.min(), 0)  # 确保包含负值
    vmax = max(data.max(), 0)  # 确保包含正值
    mid = (0 - vmin) / (vmax - vmin)  # 0值在归一化后的位置

    # 自定义颜色映射：负值(白→蓝)，0值(蓝)，正值(蓝→红)
    cdict = {
        'red':   [(0.0,   1.0, 1.0),   # 起点：白色 (R=1)
                 (mid,   0.5, 0.5),   # 零点：蓝色 (R=0.5)
                 (1.0,   1.0, 1.0)],  # 终点：红色 (R=1)
        'green': [(0.0,   1.0, 1.0),   # 起点：白色 (G=1)
                 (mid,   0.5, 0.5),   # 零点：蓝色 (G=0.5)
                 (1.0,   0.0, 0.0)],  # 终点：红色 (G=0)
        'blue':  [(0.0,   1.0, 1.0),   # 起点：白色 (B=1)
                 (mid,   1.0, 1.0),   # 零点：蓝色 (B=1)
                 (1.0,   0.0, 0.0)]   # 终点：红色 (B=0)
    }
    cmap_custom = LinearSegmentedColormap('custom_blue_red', cdict)
    norm = Normalize(vmin, vmax)

    # 绘图
    fig, axs = plt.subplots(4,4, figsize=(10, 8),
                           gridspec_kw={'wspace':0.05,'hspace':0.05})
    fig.suptitle(title, fontsize=15, y=0.92)

    for idx, ax in enumerate(axs.flat):
        im = ax.imshow(data[idx], cmap=cmap_custom, norm=norm,
                      origin='lower', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes((0.90, 0.12, 0.015, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item():.4f}")
        else:
            print(f"{name} grad: None")

def apply_gaussian_filter_with_preserved_peak(data, sigma=1.0, kernel_size=5):
    """
    应用高斯模糊同时保持峰值高度
    :param data: 输入数据 (128x128 数组)
    :param sigma: 高斯核标准差 (控制平滑程度)
    :param kernel_size: 高斯核大小 (控制影响范围)
    :return: 处理后的数据
    """
    # 创建高斯核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-distance ** 2 / (2 * sigma ** 2))

    # 归一化核（保持峰值高度）
    kernel /= kernel[center, center]

    # 应用卷积
    from scipy.ndimage import convolve
    return convolve(data, kernel, mode='constant', cval=0.0)