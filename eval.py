#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-24 12:44
# @Author : 箴澄
# @Site : 评估model文件好不好，在测试集上运行代码，看看效果
# @File : eval.py
# @Software: PyCharm
from matplotlib.colors import LinearSegmentedColormap, Normalize

from traditional.srp.SRP import plot_joint_error_heatmap
from traditional.srp.SRP4mulssl import match_sources
from load_data import AudioDoADataset, collate_fn
from model.MultiSource_3DCNN_mapNet import MultiSource3DCNNMapNet

from torch.utils.data import DataLoader
import torch
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
import cv2

def blue_red_heatmap(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    """
    生成16张 128×128 的随机三点高斯热图，4×4 网格显示。
    每图先独立高斯模糊再自身归一化，vmin=0 白色, vmax=1 纯红。
    """
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
        im = ax.imshow(data[idx], cmap=cmap, norm=norm,
                       origin='lower', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes([0.90, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs


from scipy.ndimage import maximum_filter, gaussian_filter
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import numpy as np

def simple_dbscan(points, eps=3, min_samples=2):
    """
    简单版 DBSCAN（不依赖 sklearn）
    """
    tree = cKDTree(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    labels = -1 * np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = tree.query_ball_point(points[i], eps)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        stack = neighbors.copy()

        while stack:
            j = stack.pop()
            if not visited[j]:
                visited[j] = True
                nbr = tree.query_ball_point(points[j], eps)
                if len(nbr) >= min_samples:
                    stack.extend(nbr)

            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def loc_source_position(heatmap,
                            peak_thresh=0.2,
                            neighborhood=3,
                            db_eps=5,
                            db_min_samples=1):
    """
    输入:
        heatmap: 128×128
    步骤:
        1. 最大值滤波找到局部峰
        2. 阈值筛选
        3. DBSCAN 聚类多个峰
        4. 每簇做强度加权质心
    返回:
        source_list: [(alpha, beta), ...]
    """
    # 先检查 heatmap 的值范围
    print(f"Heatmap range: [{heatmap.min()}, {heatmap.max()}]")

    # 使用相对阈值(例如最大值的 20%)
    peak_thresh = heatmap.max() * 0.2

    H = heatmap

    # 1) 局部峰值检测 (maximum filter)
    max_filt = maximum_filter(H, size=neighborhood)
    peaks = np.where((np.abs(H - max_filt) < 1e-6) & (H > peak_thresh))

    peak_points = np.stack(peaks, axis=1)   # shape (N,2)
    if len(peak_points) == 0:
        return []   # 没有声源

    # 提取峰值强度
    peak_values = H[peaks]

    # 2) DBSCAN 聚类，把多个峰值合并成一个声源
    # clustering = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(peak_points)
    # labels = clustering.labels_
    labels = simple_dbscan(peak_points, eps=db_eps, min_samples=db_min_samples)

    source_list = []

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # 噪声点略过(未归类)

        cluster_mask = labels == cluster_id
        cluster_points = peak_points[cluster_mask]   # (K,2)
        cluster_values = peak_values[cluster_mask]   # (K,)

        # 3) 强度加权质心
        w = cluster_values
        px = cluster_points[:, 0]
        py = cluster_points[:, 1]

        cx = np.sum(px * w) / np.sum(w)
        cy = np.sum(py * w) / np.sum(w)

        # 保存 (alpha, beta)
        alpha = cx - 63
        beta = cy - 63

        source_list.append((alpha, beta))

    return source_list



def evaluate_model(dataset_path, model_path='sound_model.pth'):
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiSource3DCNNMapNet().to(device)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    # 2. 加载测试数据集
    test_dataset = AudioDoADataset(root_dir=dataset_path, split="test", heatmap_label=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 3. 初始化存储
    true_azimuth = []
    true_elevation = []
    pred_azimuth = []
    pred_elevation = []

    with (torch.no_grad()):
        for x, y in test_loader:
            x = x.to(device)
            # x_test = x[:,0,:,0,0]
            # y = y.to(device)
            pred = model(x)
            # loss = heatmapLoss(pred, y)
            # logger.info(f"Test Loss: {loss.item():.6f}")
            # 计算声源位置
            for i in range(pred.shape[0]):
                pred_angle = (loc_source_position(pred[i].cpu().numpy()) )
                true_angle = (y[i].cpu().numpy())
                matches, missed_detections, false_alarms, correct_matches = match_sources(true_angle, pred_angle, 30)

                print(f"匹配情况: {matches}")
                print(f"漏检声源: {missed_detections}")
                print(f"误检声源: {false_alarms}")
                print(f"正确匹配声源: {correct_matches}")
                print()
                # logger.info(f"Missed detections: {missed_detections}")
                # logger.info(f"False alarms: {false_alarms}")
                # logger.info(f"Correct matches: {correct_matches}")
                # logger.info(f"匹配情况: {matches}")
                # 收集匹配结果
                for t_az, t_cola, p_az, p_cola in matches:
                    if t_az is not None and p_az is not None:
                        true_azimuth.append(np.radians(t_az))
                        true_elevation.append(np.radians(t_cola))
                        pred_azimuth.append(np.radians(p_az))
                        pred_elevation.append(np.radians(p_cola))

    plot_joint_error_heatmap(true_azimuth, true_elevation, pred_azimuth, pred_elevation)


if __name__ == "__main__":
    evaluate_model(dataset_path="/home/zengkehan/voice/multisource_dataset",
                   model_path="/home/zengkehan/ssl/mulsource_sound_model.pth")
