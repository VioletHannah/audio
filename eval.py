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


def loc_source_position(heatmap):
    heatmap = np.where(heatmap < 0.2, 0.0, heatmap) # 去除小于0.2的值
    heatmap_int = (heatmap * 255).astype(np.uint8)
    heatmap_int = cv2.threshold(heatmap_int, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(heatmap_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lst = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            lst.append([cx, cy])

    return np.array(lst)


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

            # row_num = 4
            # col_num = 4
            # row_pred = []
            # row_gt = []
            # for h in range(row_num):
            #     col_pred = []
            #     col_gt = []
            #     for w in range(col_num):
            #      pred_slice = pred[h * col_num + w].cpu().detach().numpy()
            #      heatmap_slice = y[h * col_num + w].cpu().detach().numpy()
            #      pred_slice = np.pad(pred_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
            #      heatmap_slice = np.pad(heatmap_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
            #      col_pred.append(pred_slice)
            #      col_gt.append(heatmap_slice)
            #     row_pred.append(np.concatenate(col_pred, axis=1))
            #     row_gt.append(np.concatenate(col_gt, axis=1))
            # result = np.concatenate(row_pred, axis=0)
            # gt = np.concatenate(row_gt, axis=0)
            # result_uint8 = (result * 255).astype('uint8')
            # cv2.imwrite("evalresult.png", result_uint8)
            # gt_uint8 = (gt * 255).astype('uint8')
            # cv2.imwrite("evalgt.png", gt_uint8)

            # true_azimuth.extend(y[:, 0].cpu().numpy().tolist())
            # true_elevation.extend(y[:, 1].cpu().numpy().tolist())

            # 预测和计算误差
            # pred_azimuth.extend(pred[:, 0].cpu().numpy().tolist())
            # pred_elevation.extend(pred[:, 1].cpu().numpy().tolist())

    plot_joint_error_heatmap(true_azimuth, true_elevation, pred_azimuth, pred_elevation)


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
                   model_path="/home/zengkehan/ssl/mulsource_sound_model.pth")
