#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-19 11:27
# @Author : 箴澄
# @File : train.py
# @Software: PyCharm

import cv2
from load_data import AudioDoADataset
from max_corr_backbone import SoundDetBackbone
from time_domain_cnn import MicArrayLocalizationNet
from ResNet_based_Net import MicArrayResNet
from MultiSource_3DCNN_mapNet import MultiSource3DCNNMapNet
from torch.utils.data import DataLoader
import torch
import numpy as np
from logger import *
from metric import *

if torch.cuda.is_available():
    device = torch.device("cuda")          # 使用 GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")           # 回退到 CPU
    print("CUDA not available, using CPU.")


datadir = "/home/zengkehan/voice/multisource_dataset"
dataset = AudioDoADataset(root_dir=datadir, split="train", n_channels=64, sample_rate=16000, duration=1.0)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# model = SoundDetBackbone()
# model = MicArrayLocalizationNet()
# model = MicArrayResNet(pretrained=True)
model = MultiSource3DCNNMapNet()
model = model.to(device)
model_path = 'mulsource_sound_model291.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
model.train()
for epoch in range(1000):
    sumloss = 0
    for inputs, heatmaps in dataloader:
        # inputs = inputs.to(device).transpose(1, 2).unsqueeze(1)
        inputs = inputs.to(device)
        heatmaps = heatmaps.to(device)
        # labels = labels.to(device)

        optimizer.zero_grad()
        pred_hm = model(inputs)

        loss = heatmapLoss(pred_hm, heatmaps)
        # loss = AngleLoss(pred, labels)
        sumloss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
        # VIS
        row_num = 4
        col_num = heatmaps.shape[0] // row_num
        row_pred = []
        row_gt = []
        for h in range(row_num):
            col_pred = []
            col_gt = []
            for w in range(col_num):
                pred_slice = pred_hm[h * col_num + w].cpu().detach().numpy()
                heatmap_slice = heatmaps[h * col_num + w].cpu().detach().numpy()
                pred_slice = np.pad(pred_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
                heatmap_slice = np.pad(heatmap_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
                col_pred.append(pred_slice)
                col_gt.append(heatmap_slice)
            row_pred.append(np.concatenate(col_pred, axis=1))
            row_gt.append(np.concatenate(col_gt, axis=1))
        result = np.concatenate(row_pred, axis=0)
        gt = np.concatenate(row_gt, axis=0)
        cv2.namedWindow("result",0)
        cv2.imshow("result", result)
        cv2.namedWindow("gt",0)
        cv2.imshow("gt", gt)
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('e'):
                exit(0)

    torch.cuda.empty_cache()
    logger.info(f"Epoch {epoch+1}, Average Loss: {sumloss/len(dataloader)}")
    if epoch % 10 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'mulsource_sound_model{epoch+1}.pth')

    scheduler.step()

