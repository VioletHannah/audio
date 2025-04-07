#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-12 21:02
# @Author : 箴澄
# @Site : 
# @File : load_data.py
# @Software: PyCharm
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
# import librosa
import glob


class AudioDoADataset(Dataset):
    def __init__(self, root_dir="G:\\audio\sin64_dataset", split="train", n_channels=16, sample_rate=16000, duration=1.0, transform=None):
        """
        参数:
        root_dir (str): 数据集根目录，包含wav和metadata文件夹
        split (str): 数据集划分，可选 "train", "val", "test"
        n_channels (int): 麦克风通道数，默认16
        sample_rate (int): 音频采样率，默认16000Hz
        duration (float): 音频时长，默认10秒
        transform (callable, optional): 音频数据的转换函数
        """
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "wavs")
        self.metadata_dir = os.path.join(root_dir, "metadata")
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.target_samples = int(sample_rate * duration)

        # 获取所有房间目录
        all_rooms = sorted(glob.glob(os.path.join(self.wav_dir, "*")))

        # 根据split划分数据集
        if split == "train":
            self.rooms = all_rooms[:int(0.8 * len(all_rooms))]
        elif split == "val":
            self.rooms = all_rooms[int(0.8 * len(all_rooms)):int(0.9 * len(all_rooms))]
        elif split == "test":
            self.rooms = all_rooms[int(0.9 * len(all_rooms)):]
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of: train, val, test")

        # 创建文件索引和对应的元数据索引
        self.room_ids = [os.path.basename(room) for room in self.rooms]
        self.metadata_files = [os.path.join(self.metadata_dir, f"{room_id}.json") for room_id in self.room_ids]

        # 检查文件是否存在
        for room, metadata_file in zip(self.rooms, self.metadata_files):
            if not os.path.exists(room):
                raise FileNotFoundError(f"Room directory not found: {room}")
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, idx):
        room_path = self.rooms[idx]
        metadata_path = self.metadata_files[idx]

        # 读取元数据JSON文件
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 获取DoA标签 (俯仰角)
        azimuth = metadata.get('source_azimuth')
        elevation = metadata.get('source_elevation')
        doa = np.array([azimuth, elevation])

        # 读取所有音频通道
        audio_files = sorted(glob.glob(os.path.join(room_path, "*.wav")))[:self.n_channels]

        if len(audio_files) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} audio files in {room_path}, but found {len(audio_files)}")

        # 将多通道音频加载到3D张量中: (n_mics_rows, n_mics_cols, n_samples)
        n_mics_rows = int(np.sqrt(self.n_channels))  # 方形阵列 (4x4 或 8x8)
        n_mics_cols = n_mics_rows

        # 初始化音频数据张量
        audio_data = np.zeros((n_mics_rows, n_mics_cols, self.target_samples), dtype=np.float32)

        # 读取每个通道的音频文件
        for i, audio_file in enumerate(audio_files):
            # 计算该通道在麦克风阵列中的位置
            row = i // n_mics_cols
            col = i % n_mics_cols

            # 读取音频
            audio, sr = sf.read(audio_file)

            # 检查采样率
            if sr != self.sample_rate:
                self.sample_rate = sr
                # audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # 处理音频长度
            if len(audio) > self.target_samples:
                # 如果音频太长，只取前面的部分
                audio = audio[:self.target_samples]
            elif len(audio) < self.target_samples:
                # 如果音频太短，用零填充
                padding = np.zeros(self.target_samples - len(audio))
                audio = np.concatenate([audio, padding])

            # 将音频数据存入相应位置
            audio_data[row, col, :] = audio

        # 将NumPy数组转换为PyTorch张量
        audio_tensor = torch.from_numpy(audio_data).float()
        doa_tensor = torch.from_numpy(doa).float()

        # 应用转换（如果有）
        if self.transform:
            audio_tensor = self.transform(audio_tensor)



        C1, C2, T = audio_tensor.shape
        audio_tensor = audio_tensor.view(C1 * C2, T)
        return audio_tensor, doa_tensor


if __name__ == "__main__":
    dataset = AudioDoADataset(
        root_dir="/home/zengkehan/voice/single64_dataset",
        split="train",
        n_channels=64,
        sample_rate=48000,
        duration=1.0
    )

    audio, doa = dataset[0]
    print(f"Audio shape: {audio.shape}")
    print(f"DoA: {doa}")

    # 测试dataloader
    from torch.utils.data import DataLoader
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    for batch_audio, batch_doa in train_loader:
        print(f"Batch audio shape: {batch_audio.shape}")
        # torch.Size([16, 64, 48000])
        print(f"Batch DoA shape: {batch_doa.shape}") # torch.Size([16, 2])
        break