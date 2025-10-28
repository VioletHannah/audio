#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-24 12:44
# @Author : 箴澄
# @Site : 评估model文件好不好，在测试集上运行代码，看看效果
# @File : eval.py
# @Software: PyCharm
import time

import numpy as np

from load_data4single import AudioDoADataset
from model.ResNet_based_Net import MicArrayResNet
from traditional.srp.SRP import plot_joint_error_heatmap
from metric import newAngleLoss

from torch.utils.data import DataLoader
import torch


def evaluate_model(dataset_path, model_path='sound_model.pth'):
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicArrayResNet().to(device)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 加载测试数据集
    test_dataset = AudioDoADataset(root_dir=dataset_path, split="test", n_channels=64)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. 初始化存储
    true_azimuth = []
    true_elevation = []
    pred_azimuth = []
    pred_elevation = []

    loss_list = []
    t1 = time.time()
    with (torch.no_grad()):
        for x, y in test_loader:
            x = x.to(device).transpose(1, 2).unsqueeze(1)
            y = y.to(device)

            true_azimuth.extend(y[:, 0].cpu().numpy().tolist())
            true_elevation.extend(y[:, 1].cpu().numpy().tolist())
            # 预测和计算误差
            pred = model(x)
            pred_azimuth.extend(pred[:, 0].cpu().numpy().tolist())
            pred_elevation.extend(pred[:, 1].cpu().numpy().tolist())

            loss = newAngleLoss(pred, y, True)
            loss_list.extend(loss.cpu().numpy().tolist())
    t2 = time.time()
    print(f"Time taken for evaluation: {t2 - t1:.2f} seconds")
    plot_joint_error_heatmap(true_azimuth, true_elevation, pred_azimuth, pred_elevation, size=0.3)

    acc0 = np.where(np.array(np.degrees(loss_list)) < 1.0)[0].shape[0] / len(loss_list)
    acc1 = np.where(np.array(np.degrees(loss_list)) < 5.0)[0].shape[0] / len(loss_list)
    acc2 = np.where(np.array(np.degrees(loss_list)) < 10.0)[0].shape[0] / len(loss_list)
    acc3 = np.where(np.array(np.degrees(loss_list)) < 15.0)[0].shape[0] / len(loss_list)
    print(f"Accuracy within 1 degree: {acc0:.2%}")
    print(f"Accuracy within 5 degrees: {acc1:.2%}")
    print(f"Accuracy within 10 degrees: {acc2:.2%}")
    print(f"Accuracy within 15 degrees: {acc3:.2%}")

if __name__ == "__main__":
    # evaluate_model(dataset_path="../voice/speech_snr_10_S",
    #                model_path="../voice/pthFile/S_100.pth")

    evaluate_model(dataset_path="../voice/speech_snr_10_L",
                   model_path="../voice/pthFile/L_220.pth")
    # evaluate_model(dataset_path="../voice/speech_snr_10",
    #                  model_path="../voice/pthFile/L_120.pth")
    # evaluate_model(dataset_path="../voice/speech_snr_30",
    #                model_path="./10snr_120.pth")
