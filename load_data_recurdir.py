#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-11-11
# @Author : 箴澄
# @Func : Load segmented multi-source SSL dataset with continuous frequency bins
# @File : load_data.py

import os
import json
import glob
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import random
import dataset_config as config
from util import azimuth_elevation_to_alpha_beta, apply_gaussian_filter_with_preserved_peak


def add_noise(audio_data, snr_db):
    """Add Gaussian white noise with target SNR (dB)."""
    signal_power = np.mean(audio_data ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), audio_data.shape)
    return audio_data + noise


def get_alpha_beta_intensity(sources, spectrums):
    """
    Convert azimuth/elevation to alpha/beta, attach continuous 128-bin intensity array.
    """
    results = []
    for i, src in enumerate(sources):
        az = src["azimuth_deg"]
        el = src["elevation_deg"]
        alpha, beta = azimuth_elevation_to_alpha_beta(az, el)
        intensity = np.array(spectrums[i]) if i < len(spectrums) else np.zeros(config.WELCH_FREQ_BINS)
        results.append((alpha, beta, intensity))
    return results


def create_heatmap_continuous(doap, grid_size=128, freq_bins=config.WELCH_FREQ_BINS):
    """
    使用连续的Welch频谱分区，从多个数据源创建二维热力图
    """
    heatmap = np.zeros((grid_size, grid_size))

    sigma_max, sigma_min = 6.0, 2.0
    freq_indices = np.arange(freq_bins)
    log_weights = np.log10(1 + freq_indices) / np.log10(1 + freq_bins)
    sigmas = sigma_max - (sigma_max - sigma_min) * log_weights

    for alpha, beta, intensity_array in doap:
        x = int(np.clip(alpha + 63, 0, grid_size - 1))
        y = int(np.clip(beta + 63, 0, grid_size - 1))

        for i, val in enumerate(intensity_array):
            if val <= 1e-6:
                continue
            band = np.zeros((grid_size, grid_size))
            band[x, y] = val
            heatmap += gaussian_filter(band, sigma=sigmas[i])

    # 标准化[0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return torch.from_numpy(heatmap).float()


class MultiSourceSSL_Dataset(Dataset):
    """
    用于连续多源声音定位的数据集。
    读取切片数据集: wavs/sample_i/segment_j/channel_k.wav + metadata/sample_i/segment_j.json
    """
    def __init__(self, root_dir, split="train", sample_rate=config.FS, heatmap_label=True, snr_db=30, aug=False):
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "wavs")
        self.meta_dir = os.path.join(root_dir, "metadata")
        self.sample_rate = sample_rate
        self.heatmap_label = heatmap_label
        self.snr_db = snr_db
        self.aug = aug
        self.indices = [0, 2, 4, 6, 9, 11, 13, 15]

        # 收集所有帧级别的JSON文件
        self.meta_files = sorted(glob.glob(os.path.join(self.meta_dir, "*", "segment_*.json")))
        if not self.meta_files:
            raise FileNotFoundError(f"No segment metadata found under {self.meta_dir}")

    def __len__(self):
        return len(self.meta_files)

    def __getitem__(self, idx):
        meta_path = self.meta_files[idx]
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        sources = meta.get("sources", [])
        welch_spectrums = meta.get("welch_spectrums", [])

        segment_dir = os.path.join(
            self.wav_dir,
            os.path.basename(os.path.dirname(meta_path)),
            os.path.splitext(os.path.basename(meta_path))[0]
        )

        selected_indices = self.indices
        n_mics = len(selected_indices)
        window_size = config.WINDOW_SIZE

        audio_data = np.zeros((n_mics, n_mics, window_size), dtype=np.float32)
        for row_idx, r in enumerate(selected_indices):
            for col_idx, c in enumerate(selected_indices):
                channel_idx = r * 16 + c
                wav_path = os.path.join(segment_dir, f"channel_{channel_idx}.wav")
                if not os.path.exists(wav_path):
                    raise FileNotFoundError(f"Missing {wav_path}")
                wav, sr = sf.read(wav_path)
                assert sr == self.sample_rate, f"采样率不匹配 {sr}!={self.sample_rate}"
                if len(wav) < window_size:
                    wav = np.pad(wav, (0, window_size - len(wav)), mode='constant')
                audio_data[row_idx, col_idx, :] = wav[:window_size]

        # 加噪声 + 标准化
        audio_data = add_noise(audio_data, self.snr_db)
        audio_data = (audio_data - np.mean(audio_data)) / (np.std(audio_data) + 1e-8)

        # (8,8,T) → (1,T,8,8) ---
        audio_tensor = torch.from_numpy(audio_data).permute(2, 0, 1).unsqueeze(0).float()

        if self.heatmap_label:
            doap = get_alpha_beta_intensity(sources, welch_spectrums)
            label = create_heatmap_continuous(doap)
        else:
            label = torch.tensor([[s["azimuth_deg"], s["elevation_deg"]] for s in sources], dtype=torch.float32)

        if self.aug:
            audio_tensor = self._augment(audio_tensor)

        return audio_tensor, label

    def _augment(self, x):
        """Random channel gain + dropout augmentation"""
        c1, c2 = x.shape[2], x.shape[3]
        # 随机增益
        gain = torch.empty(1).uniform_(0.8, 1.2)
        i, j = random.randint(0, c1 - 1), random.randint(0, c2 - 1)
        x[0, :, i, j] *= gain
        # 随机频道丢失
        if torch.rand(1) < 0.2:
            i, j = random.randint(0, c1 - 1), random.randint(0, c2 - 1)
            x[0, :, i, j] = 0.0
        return x


def collate_fn(batch):
    inputs = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return inputs, labels


if __name__ == "__main__":
    dataset = MultiSourceSSL_Dataset(
        root_dir="/home/kehan.zeng/DATA2/voice/mssl_libri_cone_segmented",
        split="train",
        aug=False
    )
    x, y = dataset[0]
    print(f"Audio tensor: {x.shape}, Heatmap: {y.shape}")
