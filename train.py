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
# model_path = 'snr_30_sound_model.pth'
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])

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
        result = pred_hm[0].cpu().detach().numpy()
        gt = heatmaps[0].cpu().detach().numpy()
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

