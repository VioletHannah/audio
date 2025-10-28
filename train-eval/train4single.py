#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-19 11:27
# @Author : 箴澄
# @File : train.py
# @Software: PyCharm
from logger import *
from load_data4single import AudioDoADataset
from model.ResNet_based_Net import MicArrayResNet
from torch.utils.data import DataLoader
from metric import *
import torch


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")          # 使用 GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")           # 回退到 CPU
        print("CUDA not available, using CPU.")

    datadir = "../voice/speech_snr_10_S"
    dataset = AudioDoADataset(root_dir=datadir, split="train", n_channels=64, sample_rate=16000, duration=1.0, aug=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MicArrayResNet(pretrained=True)
    model = model.to(device)
    model_path = '10snr_60.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    for epoch in range(1000):
        sumloss = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device).transpose(1, 2).unsqueeze(1)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            # loss = AngleLoss(pred, labels)
            loss = newAngleLoss(pred, labels)
            sumloss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            del inputs, labels, pred

        torch.cuda.empty_cache()
        logger.info(f"Epoch {epoch+1}, Average Loss: {sumloss/len(dataloader)}")
        # print(f"Epoch {epoch+1}, Average Loss: {sumloss/len(dataloader)}")

        if epoch % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'S_{epoch}.pth')

        scheduler.step()

if __name__ == "__main__":
    main()