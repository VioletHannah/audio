#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-12 21:02
# @Author : 箴澄
# @Site : 
# @File : load_data.py
# @Software: PyCharm
import os
import json
import random
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import soundfile as sf
import glob
import math
from util import apply_gaussian_filter_with_preserved_peak, azimuth_elevation_to_alpha_beta, heatmap_plot


def get_alpha_beta(sources):
    result_list = []
    for source in sources:
        # 提取azimuth_deg和elevation_deg
        azimuth_deg = source["azimuth_deg"]
        elevation_deg = source["elevation_deg"]

        # 计算alpha和beta
        alpha, beta = azimuth_elevation_to_alpha_beta(azimuth_deg, elevation_deg)
        result_list.append((alpha, beta))

    return result_list

def get_alpha_beta_intensity(sources, intensities_list):
    result_list = []
    for i, source in enumerate(sources):
        # 提取azimuth_deg和elevation_deg计算alpha和beta
        theta = source["azimuth_deg"]
        phi = source["elevation_deg"]
        alpha, beta = azimuth_elevation_to_alpha_beta(theta, phi)

        # 提取各个频段强度值
        band_intensity = intensities_list[i] if i < len(intensities_list) else [0.0]*5
        result_list.append((alpha, beta, band_intensity))
    return result_list

def collate_fn(batch):
    # 分离数据和标签
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # 堆叠数据（假设输入形状一致）
    inputs = torch.stack(inputs, dim=0)
    # 标签保持为列表，每个元素是原始形状
    return inputs, targets

def add_noise(audio_data, SNR):
    signal_power = np.mean(audio_data ** 2)
    noise_power = signal_power / (10 ** (SNR / 10))
    if noise_power == 0:
        noise_power = 0.001
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, audio_data.shape)
    audio_data += noise
    return  audio_data


class AudioDoADataset(Dataset):
    def __init__(self, root_dir="G:\\audio\sin64_dataset", split="train", n_channels=64, sample_rate=48000, duration=1.0, heatmap_label=True, augm = False):
        """
        root_dir (str): 数据集根目录，包含wav和metadata文件夹
        split (str): 数据集划分，可选 "train", "val", "test"
        n_channels (int): 麦克风通道数，默认16
        sample_rate (int): 音频采样率，默认16000Hz
        duration (float): 音频时长，默认10秒
        transform (callable, optional): 音频数据的转换函数
        heatmap_label (bool): 是否使用热力图标签，默认True
        augm (bool): 是否进行数据增强，默认True
        """
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "wavs")
        self.metadata_dir = os.path.join(root_dir, "metadata")
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.duration = duration
        self.heatmap_label = heatmap_label
        self.augm = augm

        self.window_samples = 16000
        # self.target_samples = int(sample_rate * duration)
        self.selected_indices = [0, 2, 4, 6, 9, 11, 13, 15]  # 对应1,3,5,7,10,12,14,16
        self.n_mics = len(self.selected_indices)

        self.center_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        self.sigmas = self._compute_sigmas(self.center_freqs)
        self.kernel_sizes = [max(3, min(int(6 * sigma) + 1, 21)) for sigma in self.sigmas]

        # 获取所有房间目录
        all_rooms = sorted(glob.glob(os.path.join(self.wav_dir, "*")))

        # 根据split划分数据集
        if split == "train":
            self.rooms = all_rooms[:int(0.9 * len(all_rooms))]
        elif split == "val":
            self.rooms = all_rooms
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

    @staticmethod
    def _compute_sigmas(center_freqs, min_sigma=2.0, max_sigma=6.0):
        """
        预计算每个频段对应的 sigma 值

        Args:
            center_freqs: 中心频率列表
            min_sigma: 高频对应的最小 sigma
            max_sigma: 低频对应的最大 sigma

        Returns:
            sigmas: 每个频段的 sigma 值列表
        """
        min_freq, max_freq = center_freqs[0], center_freqs[-1]
        sigmas = []

        for freq in center_freqs:
            # 在对数尺度上线性插值
            log_freq = np.log10(freq)
            log_min = np.log10(min_freq)
            log_max = np.log10(max_freq)

            # 归一化到 0-1 范围
            normalized = (log_freq - log_min) / (log_max - log_min)

            # 映射到 sigma 范围（高频对应小 sigma，低频对应大 sigma）
            sigma = max_sigma - normalized * (max_sigma - min_sigma)
            sigmas.append(sigma)

        return sigmas

    def __getitem__(self, idx):
        room_path = self.rooms[idx]
        metadata_path = self.metadata_files[idx]

        # 读取元数据JSON文件
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 获取DoA标签 (俯仰角)
        sources = metadata.get('sources', [])
        band_intensities = metadata.get('intensities', [])  # 多频段强度信息

        # 16x16阵列共256通道，筛选出需要的8x8通道
        selected_channels = []
        for r in self.selected_indices:
            for c in self.selected_indices:
                channel_idx = r * 16 + c
                selected_channels.append(channel_idx)

        # 初始化8x8音频张量
        audio_data = np.zeros((self.n_mics, self.n_mics, self.window_samples), dtype=np.float64)

        # 读取并填充选中的通道数据
        for row_idx, r in enumerate(self.selected_indices):
            for col_idx, c in enumerate(self.selected_indices):
                channel = r * 16 + c
                audio_file = os.path.join(room_path, f"channel_{channel}.wav")
                # 读取48kHz原始音频
                audio, sr = sf.read(audio_file)
                assert sr == self.sample_rate, f"采样率不匹配: {sr} != {self.sample_rate}"
                # 截取16000样本窗口
                audio_data[row_idx, col_idx, :] = audio[len(audio)-self.window_samples:]

        # 添加噪音 - 可选
        audio_data = add_noise(audio_data, SNR = 60)

        # 数据标准化
        audio_data = (audio_data - np.mean(audio_data)) / (np.std(audio_data) + 1e-9)

        # TODO: 测试Audio数据
        # print(f"Audio data shape: {audio_data.shape}, min: {np.min(audio_data):.6f}, max: {np.max(audio_data):.6f}, mean: {np.mean(audio_data):.6f}, std: {np.std(audio_data):.6f}")

        # 处理标签
        doap = get_alpha_beta_intensity(sources, band_intensities)

        # 转换为网络输入形状
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = audio_tensor.permute(2, 0, 1).unsqueeze(0) # (R, C, T) -> (1, T, C1, C2)

        if self.heatmap_label == True:
            # heatmap_tensor = create_heatmap(doa_tensor)
            # heatmap_tensor = create_heatmap(doap_tensor, grid_size=128, sigma=2)
            heatmap_tensor = create_heatmap_multiband(doap, grid_size=128)
        else:
            result = []
            for source in sources:
                theta = source["azimuth_deg"]
                phi = source["elevation_deg"]
                result.append((theta, phi))
            heatmap_tensor = torch.tensor(result).float()

        # 数据增强
        if self.augm:
            audio_tensor = self._augment(audio_tensor)
        return audio_tensor, heatmap_tensor

    def _augment(self, x):
        """
        数据增强函数
        :param x: Tensor of shape (1, T, C1, C2)
        :return: x: 增强后的音频张量
        """
        c1, c2 = x.shape[2], x.shape[3]

        # 1 对某一通道随机增益(0.8, 1.2)
        gain = torch.empty(1).uniform_(0.8, 1.2)
        i, j = random.randint(0, c1 - 1), random.randint(0, c2 - 1)
        x[0, :, i, j] *= gain

        # 2 随机通道丢弃（模拟麦克风故障）
        if torch.rand(1) < 0.2:
            n_drop = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            for _ in range(n_drop):
                # 随机选择一个通道进行丢弃
                i, j = random.randint(0, c1 - 1), random.randint(0, c2 - 1)
                x[0, :, i, j] = 0.0

        return x

    def _create_heatmap_multiband(self, doap, grid_size=128):
        """
        使用预计算的 sigma 和 kernel_size 创建热图

        Args:
            doap: 包含 (alpha, beta, intensity_list) 的列表
            grid_size: 热图尺寸

        Returns:
            热图张量
        """
        heatmap = np.zeros((grid_size, grid_size))

        # 遍历每个声源
        for point in doap:
            alpha, beta, intensity_list = point
            x = int(np.clip(alpha + 63, 0, grid_size - 1))
            y = int(np.clip(beta + 63, 0, grid_size - 1))

            # 遍历每个频段（使用预计算的参数）
            for band_idx, intensity in enumerate(intensity_list):
                sigma = self.sigmas[band_idx]

                # 创建当前频段的热力图
                band_heatmap = np.zeros((grid_size, grid_size))
                band_heatmap[x, y] = intensity

                # 应用高斯滤波
                band_heatmap = gaussian_filter(band_heatmap, sigma=sigma)

                # 叠加到总热力图
                heatmap += band_heatmap

        # 归一化热力图到 0-1 范围
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return torch.from_numpy(heatmap).float()


# doa: 一个[source num, 2]的二维数组
# doap: 一个[source num, 3]的二维数组，包含alpha, beta和intensity
def create_heatmap(doa, grid_size=128, sigma=4, kernel_size=None):
    # 初始化热力图矩阵
    heatmap = np.zeros((grid_size, grid_size))
    # 自动计算合理的核大小
    if kernel_size is None:
        kernel_size = int(6 * sigma) + 1  # 经验公式：核大小 ≈ 6×σ
        # kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # 确保为奇数
        kernel_size = max(3, min(kernel_size, 21))  # 限制在3-21之间

    # 遍历每个样本
    # sources = doa.cpu().numpy()
    sources = doa.numpy() if isinstance(doa, torch.Tensor) else doa
    for point in sources:  # 遍历每个声源
        alpha, beta, intensity = point
        # 坐标映射（假设原始范围是 [-63, 64)）
        x = int(np.clip(alpha + 63, 0, grid_size - 1))  # 防止越界
        y = int(np.clip(beta + 63, 0, grid_size - 1))
        # print(intensity)
        # 在附近位置都加上强度值
        heatmap[x, y] = intensity / 100 # 直接累加强度值
        # heatmap[x, y] = np.log10(intensity / (2e-5)) / 6 if intensity > (2e-5) else 0 # 缩放到 [0,1]

    # 对每个样本单独应用高斯滤波
    # heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap = apply_gaussian_filter_with_preserved_peak(heatmap, sigma=sigma, kernel_size=kernel_size)

    return torch.from_numpy(heatmap).float()

def create_heatmap_multiband(doap, grid_size=128, center_freqs=[31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]):
    # 初始化热力图矩阵
    heatmap = np.zeros((grid_size, grid_size))
    num_bands = len(center_freqs)

    # 为每个频段计算sigma值（高频使用较小的sigma，低频使用较大的sigma）
    # 使用对数尺度映射频率到sigma
    min_freq, max_freq = center_freqs[0], center_freqs[-1]

    # sigma范围：高频2.0，低频6.0
    min_sigma, max_sigma = 2.0, 6.0

    # 计算每个频段的sigma
    sigmas = []
    for freq in center_freqs:
        # 在对数尺度上线性插值
        log_freq = np.log10(freq)
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)

        # 归一化到0-1范围
        normalized = (log_freq - log_min) / (log_max - log_min)

        # 映射到sigma范围（高频对应小sigma，低频对应大sigma）
        sigma = max_sigma - normalized * (max_sigma - min_sigma)
        sigmas.append(sigma)

    # 遍历每个声源
    for point in doap:
        alpha, beta, intensity_list = point
        x = int(np.clip(alpha + 63, 0, grid_size - 1))
        y = int(np.clip(beta + 63, 0, grid_size - 1))

        # 遍历每个频段
        for band_idx, intensity in enumerate(intensity_list):
            sigma = sigmas[band_idx]
            kernel_size = int(6 * sigma) + 1
            kernel_size = max(3, min(kernel_size, 21))

            # 创建当前频段的热力图
            band_heatmap = np.zeros((grid_size, grid_size))
            band_heatmap[x, y] = intensity

            # 应用高斯滤波
            band_heatmap = gaussian_filter(band_heatmap, sigma=sigma)

            # TODO: 测试代码 - 可视化每个频段的热力图
            # heatmap_plot(band_heatmap, title=f"Band {center_freqs[band_idx]} Hz Heatmap", absflag=True)

            # 叠加到总热力图
            heatmap += band_heatmap
    # TODO: 测试代码 - 可视化叠加后的热力图
    # heatmap_plot(heatmap, title="Combined Heatmap")

    # 归一化热力图到0-1范围
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    return torch.from_numpy(heatmap).float()


if __name__ == "__main__":
    dataset = AudioDoADataset(
        root_dir="/home/kehan.zeng/DATA2/voice/mssl_libri",
        split="train",
        n_channels=256,
        sample_rate=48000
    )

    audio, doa = dataset[0]
    print(f"Audio shape: {audio.shape}")
    print(f"DoA shape: {doa.shape}")

    # 测试dataloader
    from torch.utils.data import DataLoader
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch_audio, batch_doa in train_loader:
        print(f"Batch audio shape: {batch_audio.shape}")
        print(f"Batch DoA shape: {batch_doa.shape}")
        heatmap_plot(batch_doa[0].numpy(), title="Sample Heatmap")
        break
"""
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for batch_audio, batch_doa in train_loader:
        print(f"Batch audio shape: {batch_audio.shape}")
        # torch.Size([16, 64, 48000])
        print(f"Batch DoA: {batch_doa}") # torch.Size([16, 2])
        # 计算热力图
        heatmap = create_heatmap_multiband(batch_doa)
        print(f"Heatmap shape: {heatmap.shape}")
        break
"""
