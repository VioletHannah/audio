#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-24 12:44
# @Author : 箴澄
# @Site : 评估model文件好不好，在测试集上运行代码，看看效果
# @File : eval.py
# @Software: PyCharm
from util import blue_red_heatmap, print_gradients, get_logger, blue_red_heatmap_old
from load_data import AudioDoADataset
from metric import heatmapLoss
# from model.MultiSource_3DCNN_mapNet_revise import MultiSource3DCNNMapNetRevise
from model.MultiSource_3DCNN_mapNet import MultiSource3DCNNMapNet

from torch.utils.data import DataLoader
import torch

# 忽略Matplotlib的警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def evaluate_model(dataset_path, model_path='sound_model.pth', splitflag="test"):
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MultiSource3DCNNMapNetRevise().to(device)
    model = MultiSource3DCNNMapNet().to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    logger = get_logger("./eval_vis.log")

    # 2. 加载测试数据集
    test_dataset = AudioDoADataset(root_dir=dataset_path, split=splitflag)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

    # 3. 初始化存储
    # true_azimuth = []
    # true_elevation = []
    # pred_azimuth = []
    # pred_elevation = []
    # sumloss = []
    with (torch.no_grad()):
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = heatmapLoss(pred, y)
            logger.info(f"Test Loss: {loss.item():.6f}")
            blue_red_heatmap_old(y[0:16].cpu().detach().numpy() * 5, "gt")
            blue_red_heatmap_old(pred[0:16].cpu().detach().numpy() * 5, "pred")

            # sumloss.append(loss.item())

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
def plot_gt(dataset_path):
    dataset = AudioDoADataset(root_dir=dataset_path, split="val")
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for _, y in test_loader:
        blue_red_heatmap_old(y[0:16].cpu().detach().numpy(), "gt")

if __name__ == "__main__":
    evaluate_model(dataset_path="/home/kehan.zeng/DATA2/voice/multisource_with_freq_analysis",
                   model_path="/home/kehan.zeng/DATA1/param/freq/freq_1990.pth", splitflag="test")
    # plot_gt("/home/kehan.zeng/DATA2/voice/multisource_normalized")

    # evaluate_model(dataset_path="/home/zengkehan/voice/multisource4eval_2",
    #                model_path="/home/zengkehan/ssl/mulsource_newloss_model791.pth")
    # evaluate_model(dataset_path="/home/zengkehan/voice/multisource4eval_3",
    #                model_path="/home/zengkehan/ssl/mulsource_newloss_model791.pth")