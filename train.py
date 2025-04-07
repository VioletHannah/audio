#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-19 11:27
# @Author : 箴澄
# @File : train.py
# @Software: PyCharm

from load_data import AudioDoADataset
from max_corr_backbone import SoundDetBackbone
from torch.utils.data import DataLoader
import torch
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")          # 使用 GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")           # 回退到 CPU
    print("CUDA not available, using CPU.")

import torch


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

datadir = "/home/zengkehan/voice/single64_dataset"
dataset = AudioDoADataset(root_dir=datadir, split="train", n_channels=64, sample_rate=48000, duration=10.0)
# dataset = AudioDoADataset(root_dir="G:\\audio\\sin64_dataset", split="train")
model = SoundDetBackbone()
model = model.to(device)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    sumloss = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        pred = model(inputs)
        # pred = output
        # for i in output:
        #     pred = i
        #     break
        # theta_loss, phi_loss = AngleLoss(pred, y)
        # loss = (theta_loss + phi_loss).mean()
        loss = AngleLoss(pred, labels)
        sumloss += loss.item()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        del inputs, labels, pred, loss

    torch.cuda.empty_cache()
    print(f"Epoch {epoch+1}, Average Loss: {sumloss/len(dataloader)}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'sound_model.pth')