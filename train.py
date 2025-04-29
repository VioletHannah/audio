#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-19 11:27
# @Author : 箴澄
# @File : train.py
# @Software: PyCharm

from load_data import AudioDoADataset
from load_data import collate_fn
from max_corr_backbone import SoundDetBackbone
from time_domain_cnn import MicArrayLocalizationNet
from ResNet_based_Net import MicArrayResNet
from MultiSource_3DCNN_mapNet import MultiSource3DCNNMapNet
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

if torch.cuda.is_available():
    device = torch.device("cuda")          # 使用 GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")           # 回退到 CPU
    print("CUDA not available, using CPU.")

def create_heatmap(doa, grid_size=128, sigma=2):
    # doa: 一个列表，内有B个Tensor，每个Tensor的形状为[source num, 2]
    batch_size = len(doa)
    # 初始化热力图矩阵
    heatmap = np.zeros((batch_size, grid_size, grid_size))
    # 遍历每个样本
    for b in range(batch_size):
        sources = doa[b].cpu().numpy()
        for point in sources:  # 遍历每个声源
            alpha, beta = point
            # 坐标映射（假设原始范围是 [-63, 64)）
            x = int(np.clip(alpha + 63, 0, grid_size - 1))  # 防止越界
            y = int(np.clip(beta + 63, 0, grid_size - 1))
            heatmap[b, x, y] += 1
        # 归一化
        heatmap[b] = heatmap[b] / np.max(heatmap[b]) if np.max(heatmap[b]) > 0 else 0
        # 对每个样本单独应用高斯滤波
        heatmap[b] = gaussian_filter(heatmap[b], sigma=sigma)

    return torch.from_numpy(heatmap).float()


def heatmapLoss(pred, target):
    # pred: [B, 128, 128]，target: [B, 128, 128]
    mse = torch.nn.MSELoss()
    # 计算均方误差损失
    loss = mse(pred, target)
    return torch.mean(loss)

def AngleLoss(pred, target):
    # pred: [B, 2]（弧度），target: [B, 2]（弧度）
    theta_pred, phi_pred = pred[:, 0], pred[:, 1]
    theta_target, phi_target = target[:, 0], target[:, 1]

    # 处理theta的360°周期性（正确周期为2π）
    theta_diff = torch.abs(theta_pred - theta_target) % (2 * torch.pi)
    theta_loss = torch.min(theta_diff, 2 * torch.pi - theta_diff)  # 修正周期为2π

    phi_loss = torch.abs(phi_pred - phi_target)

    # 将弧度转换为角度
    deg_factor = 180.0 / torch.pi
    theta_loss_deg = theta_loss * deg_factor
    phi_loss_deg = phi_loss * deg_factor

    # 返回角度损失的平均值
    return (theta_loss_deg + phi_loss_deg).mean()

datadir = "/home/zengkehan/voice/multisource_dataset"
dataset = AudioDoADataset(root_dir=datadir, split="train", n_channels=64, sample_rate=16000, duration=1.0)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True,  collate_fn=collate_fn)

# model = SoundDetBackbone()
# model = MicArrayLocalizationNet()
# model = MicArrayResNet(pretrained=True)
model = MultiSource3DCNNMapNet()
model = model.to(device)
# model_path = 'snr_30_sound_model.pth'
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
for epoch in range(1000):
    sumloss = 0
    for inputs, labels in dataloader:
        # inputs = inputs.to(device).transpose(1, 2).unsqueeze(1)
        inputs = inputs.to(device)
        # labels = labels.to(device)
        gt_hm = create_heatmap(labels).to(device)

        optimizer.zero_grad()
        pred_hm = model(inputs)

        loss = heatmapLoss(pred_hm, gt_hm)
        # loss = AngleLoss(pred, labels)
        sumloss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        del inputs, labels, pred_hm, gt_hm

    torch.cuda.empty_cache()
    print(f"Epoch {epoch+1}, Average Loss: {sumloss/len(dataloader)}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'mulsource_sound_model.pth')

    scheduler.step()