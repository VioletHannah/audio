#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-12 17:11
# @Author : 箴澄
# @SRP : Steered Response Power with Phase Transform
# @File : SRP.py
# @Software: PyCharm
import json
import time

import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import os
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from traditional.tdoa.TDOA import calculate_spatial_angle

# from eval import plot_polar_heatmap

def plot_joint_error_heatmap(true_azimuth, true_elevation, pred_azimuth, pred_elevation, snr=30, size=0.2):
    """
    绘制方位角与俯仰角联合误差热力图

    参数：
      true_azimuth    : 真实方位角列表（弧度）
      true_elevation  : 真实仰角列表（弧度）
      pred_azimuth    : 预测方位角列表（弧度）
      pred_elevation  : 预测仰角列表（弧度）
    """
    # 转换为角度
    true_az_deg = np.degrees(true_azimuth) % 360
    pred_az_deg = np.degrees(pred_azimuth) % 360
    true_el_deg = np.degrees(true_elevation)
    pred_el_deg = np.degrees(pred_elevation)

    # 计算周期性方位角误差（映射到[-180, 180)）
    az_errors = pred_az_deg - true_az_deg
    az_errors = (az_errors + 180) % 360 - 180

    # 计算仰角误差（限制在[-90, 90]）
    el_errors = pred_el_deg - true_el_deg
    el_errors = np.clip(el_errors, -90, 90)

    # 设置分箱参数
    az_bins = np.linspace(-180, 180, 73)  # 每5度一个分箱
    el_bins = np.linspace(-90, 90, 37)  # 每5度一个分箱

    # 计算二维直方图
    hist, xedges, yedges = np.histogram2d(az_errors, el_errors, bins=[az_bins, el_bins], density=False)

    # 创建图像
    plt.figure(figsize=(12, 6), dpi=200)

    # 设置字体大小
    plt.rcParams.update({'font.size': 15})

    # 设置坐标轴
    ax = plt.gca()

    # 绘制热力图（使用对数归一化）
    im = ax.imshow(
        hist.T,
        extent=[-180, 180, -90, 90],
        origin='lower',
        aspect='auto',
        cmap='jet',  # 蓝-青-黄-红渐变
        norm=LogNorm(vmin=1, vmax=hist.max())  # 对数色标
    )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Number of Samples', rotation=270, labelpad=15)

    # 设置坐标轴
    ax.set_xlabel('Azimuth Error (degrees)\n(Prediction - Truth)')
    ax.set_ylabel('Elevation Error (degrees)\n(Prediction - Truth)')

    # 绘制辅助网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))

    # 平均误差标注
    mean_az_error = np.mean(np.abs(az_errors))
    mean_el_error = np.mean(np.abs(el_errors))
    stats_text = (
        # f'Total Samples: {len(az_errors):,}\n'
        # f'SNR: {snr}dB\n'
        f'MicArray Size: {size}m\n'
        f'Mean Azimuth Error: {mean_az_error:.2f}°\n'
        f'Mean Elevation Error: {mean_el_error:.2f}°'
    )
    ax.text(0.63, 0.95, stats_text,
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return mean_az_error, mean_el_error


def stft(signal, frame_size, hop_size, nfft, window=None):
    """
    参数：
      signal     : 一维信号数组
      frame_size : 帧长
      hop_size   : 帧移
      nfft       : FFT 点数
      window     : 窗函数数组，默认使用 np.hanning(frame_size)

    返回：
      stft_matrix: 复数数组，形状为 (n_frames, nfft//2+1)，每一行对应一帧做 rfft 后的结果
    """
    if window is None:
        window = np.hanning(frame_size)

    # 计算帧数（只考虑完整帧）
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    stft_matrix = np.empty((n_frames, nfft // 2 + 1), dtype=complex)

    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start: start + frame_size]
        frame_windowed = frame * window
        stft_matrix[i, :] = np.fft.rfft(frame_windowed, n=nfft)

    return stft_matrix


def load_wav_files_to_freqdomain(folder, num_channels=64):
    """
    加载一个room中的文件，要求文件名为 channel_0.wav, channel_2.wav, ..., channel_15.wav
    返回频域信号和采样率
    """
    signals = []
    fs = None

    # 逐个加载通道文件
    for i in range(num_channels):
        filename = os.path.join(folder, f"channel_{i}.wav")
        audio, fs_tmp = sf.read(filename)
        if fs is None:
            fs = fs_tmp
        elif fs != fs_tmp:
            raise ValueError("所有通道必须具有相同的采样率")
        signals.append(audio)
    signals = np.array(signals)
    # 若各通道采样点数不一致，取最短长度
    min_length = min([len(s) for s in signals])
    signals = signals[:, :min_length]

    # if signals.shape[1] < nfft:
    #     # 填充信号至nfft长度
    #     padded_signals = np.zeros((signals.shape[0], nfft))
    #     padded_signals[:, :signals.shape[1]] = signals
    #     signals = padded_signals
    # signals = signals[:, :nfft] # 取nfft个采样点

    nfft = 1024  # FFT点数
    M = signals.shape[0]  # 通道数
    win = np.hanning(nfft)  # 窗函数
    hop = nfft // 2  # 帧移
    freq_signals = []  # 频域信号列表

    # 对每个通道信号进行STFT
    for m in range(M):
        # 对每个通道信号进行STFT，返回复数频域信号[nfft//2+1, n_frames]
        stft_result = stft(signals[m, :], nfft, hop, nfft, window=win)
        freq_signals.append(stft_result)

    freq_signals = np.array(freq_signals)
    freq_signals = np.transpose(freq_signals, (0, 2, 1))  # 转置为 [num_channels, n_frames, nfft//2+1]

    return freq_signals, fs


def main():
    # azimuth_error = []
    # cola_error = []
    true_cola = []
    true_az = []
    pred_az = []
    pred_cola = []

    # 设置参数
    room_dimension = np.array([50, 50, 3])  # 房间尺寸
    mic_positions = np.zeros((3, 64))
    spacing = 0.2 / (8 - 1)  # 麦克风间距
    mic_num_per_line = 8
    offset = (mic_num_per_line - 1) / 2  # 中心对称偏移量
    for i in range(mic_num_per_line):
        for j in range(mic_num_per_line):
            index = i * mic_num_per_line + j
            mic_positions[0, index] = room_dimension[0] / 2 + (i - offset) * spacing
            mic_positions[1, index] = room_dimension[1] / 2 + (j - offset) * spacing

    # azimuth：0°～360°， colatitude = 90° - elevation， 0～π/2
    azimuth_search = np.radians(np.linspace(0, 360, 360))
    colatitude_search = np.radians(np.linspace(0, 90, 91))

    for i in range(100):
        ####################################################
        base_folder = "../voice/speech_no_noise"  # 数据集路径
        ####################################################
        folder = os.path.join(base_folder, "wavs", f"sample_{i}")
        try:
            # 计算频域信号
            freqsignals, fs = load_wav_files_to_freqdomain(folder, 64)
        except:
            continue
        doa = pra.doa.srp.SRP(mic_positions, fs, nfft=1024,
                              azimuth=azimuth_search, colatitude=colatitude_search, dim=3)
        t1 = time.time()
        doa.locate_sources(freqsignals)
        t2 = time.time()
        print(f"第 {i + 1} 个样本定位耗时: {t2 - t1:.2f}秒")

        # 获取估计结果（单位为弧度）
        estimated_azimuth = doa.azimuth_recon[0]
        estimated_colatitude = doa.colatitude_recon[0]

        # 真实方位角和余纬度
        label_file = os.path.join(base_folder, "metadata", f"sample_{i}.json")
        with open(label_file, 'r') as f:
            metadata = json.load(f)
        true_azimuth = metadata['source_azimuth']
        true_elevation = metadata['source_elevation']
        true_colatitude = np.pi / 2 - true_elevation

        # 存储真实值和估计值
        true_az.append(true_azimuth)
        true_cola.append(true_colatitude)
        pred_az.append(estimated_azimuth)
        pred_cola.append(estimated_colatitude)

        # 计算误差，考虑角度周期性
        # def angle_error(est, true):
        #     err = abs(est - true)
        #     return err if err <= 180 else 360 - err
        #
        # error_azimuth = angle_error(estimated_azimuth, true_azimuth)
        # error_colatitude = angle_error(estimated_colatitude, true_colatitude)

        print("真实方位角: {:.2f}°".format(np.degrees(true_azimuth)))
        print("估计方位角: {:.2f}°".format(np.degrees(estimated_azimuth)))
        print("方位角误差: {:.2f}°".format(np.degrees(estimated_azimuth - true_azimuth)))
        print("真实余纬度: {:.2f}°  (对应真实仰角: {:.2f}°)".format(np.degrees(true_colatitude),
                                                                    np.degrees(true_elevation)))
        print("估计余纬度: {:.2f}°  (对应估计仰角: {:.2f}°)".format(np.degrees(estimated_colatitude),
                                                                    np.degrees(np.pi / 2 - estimated_colatitude)))
        print("余纬度误差: {:.2f}°".format(np.degrees(estimated_colatitude - true_colatitude)))
        print(f"已处理第 {i + 1} 个样本")
        print("--------------------------------------------------")
        # azimuth_error.append(error_azimuth)
        # cola_error.append(error_colatitude)

    # 可视化
    ame, cme = plot_joint_error_heatmap(true_az, true_cola, pred_az, pred_cola)
    print(f"平均方位角误差: {ame}°")
    print(f"平均余纬度误差: {cme}°")
    error = []
    for i in range(len(true_az)):
        error.append(calculate_spatial_angle(pred_az[i], pred_cola[i], true_az[i], true_cola[i]))
    acc1 = np.where(np.degrees(error) < 5)[0].shape[0] / len(error)
    print(f"5°内定位精度: {acc1:.2%}")
    acc2 = np.where(np.degrees(error) < 10)[0].shape[0] / len(error)
    print(f"10°内定位精度: {acc2:.2%}")
    acc3 = np.where(np.degrees(error) < 15)[0].shape[0] / len(error)
    print(f"15°内定位精度: {acc3:.2%}")

    # print(f"平均方位角误差: {np.mean(azimuth_error)}°")
    # print(f"平均余纬度误差: {np.mean(cola_error)}°")

if __name__ == "__main__":
    main()