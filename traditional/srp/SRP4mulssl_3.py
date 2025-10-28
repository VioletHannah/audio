#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/21 08:52
# @Author : 箴澄
# @Func：
# @File : SRP4mulssl.py
# @Software: PyCharm
# @SRP : Steered Response Power with Phase Transform
import json
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from logger import *

def plot_scatter(true_azimuth, true_colatitude, pred_azimuth, pred_colatitude):
    """
    在散点图中绘制真值与预测值的方位角和余纬度对比

    参数：
      true_azimuth    : 真实方位角列表（弧度）
      true_colatitude : 真实余纬度列表（弧度）
      pred_azimuth    : 预测方位角列表（弧度）
      pred_colatitude : 预测余纬度列表（弧度）
    """
    # 转换单位为角度
    true_az_deg = np.degrees(true_azimuth)
    true_cola_deg = np.degrees(true_colatitude)
    pred_az_deg = np.degrees(pred_azimuth)
    pred_cola_deg = np.degrees(pred_colatitude)

    # 创建散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(true_az_deg, true_cola_deg, c='red', s=20, label='True', alpha=0.7)
    ax.scatter(pred_az_deg, pred_cola_deg, c='blue', s=20, marker='x', label='Predicted', alpha=0.7)

    # 绘制连接线
    for t_az, t_cola, p_az, p_cola in zip(true_az_deg, true_cola_deg, pred_az_deg, pred_cola_deg):
        ax.plot([t_az, p_az], [t_cola, p_cola], color='gray', alpha=0.3, linewidth=0.5)

    # 设置坐标轴标签和标题
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Colatitude (°)')
    ax.set_title('True vs Predicted Direction Comparison\n(Azimuth & Elevation)')
    plt.legend(loc='upper right')

    # 添加误差统计标注
    az_errors = np.abs(true_az_deg - pred_az_deg)
    az_errors = np.minimum(az_errors, 360 - az_errors)
    el_errors = np.abs((90 - true_cola_deg) - (90 - pred_cola_deg))

    stats_text = f'Mean Azimuth Error: {np.mean(az_errors):.2f}°\n' \
                 f'Mean Elevation Error: {np.mean(el_errors):.2f}°'
    plt.annotate(stats_text, xy=(0.1, 0.9), xycoords='axes fraction', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_joint_error_heatmap(true_azimuth, true_elevation, pred_azimuth, pred_elevation):
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
    plt.figure(figsize=(12, 6))
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
    ax.set_xlabel('Azimuth Error (degrees)\n(Prediction - Truth)', fontsize=12)
    ax.set_ylabel('Elevation Error (degrees)\n(Prediction - Truth)', fontsize=12)
    ax.set_title('Joint Azimuth-Elevation Error Distribution', fontsize=14, pad=20)

    # 绘制辅助网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))

    # 平均误差标注
    mean_az_error = np.mean(np.abs(az_errors))
    mean_el_error = np.mean(np.abs(el_errors))
    stats_text = (
        f'Total Samples: {len(az_errors):,}\n'
        f'Mean Azimuth Error: {mean_az_error:.2f}°\n'
        f'Mean Elevation Error: {mean_el_error:.2f}°'
    )
    ax.text(1.15, 1.10, stats_text,
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

    nfft = 1024 # FFT点数
    M = signals.shape[0] # 通道数
    win = np.hanning(nfft) # 窗函数
    hop = nfft // 2 # 帧移
    freq_signals = [] # 频域信号列表

    # 对每个通道信号进行STFT
    for m in range(M):
        # 对每个通道信号进行STFT，返回复数频域信号[nfft//2+1, n_frames]
        stft_result = stft(signals[m, :], nfft, hop, nfft, window=win)
        freq_signals.append(stft_result)

    freq_signals = np.array(freq_signals)
    freq_signals = np.transpose(freq_signals, (0, 2, 1)) # [M, F, S]

    return freq_signals, fs


def match_sources(true_angles, pred_angles, max_error=30):
    """
    使用最近邻匹配真实声源和预测声源
    true_angles: 真实角度列表 [(az1, cola1), ...]
    pred_angles: 预测角度列表 [(az2, cola2), ...]
    max_error: 最大允许匹配误差（度）
    返回匹配后的角度对列表，漏检数量，错检数量，定位正确数量
    """
    matched = []
    used_pred = set()
    correct_matches = 0

    # 处理真实声源为空的情况
    if len(true_angles) == 0:
        for p_az, p_cola in pred_angles:
            matched.append((None, None, p_az, p_cola))
        missed_detections = 0
        false_alarms = len(pred_angles)
        return matched, missed_detections, false_alarms, correct_matches

    # 处理预测声源为空的情况
    if len(pred_angles) == 0:
        for t_az, t_cola in true_angles:
            matched.append((t_az, t_cola, None, None))
        missed_detections = len(true_angles)
        false_alarms = 0
        return matched, missed_detections, false_alarms, correct_matches

    for t_az, t_cola in true_angles:
        min_dist = float('inf')
        best_idx = -1
        for p_idx, (p_az, p_cola) in enumerate(pred_angles):
            if p_idx in used_pred:
                continue
            az_error = min(abs(p_az - t_az), 360 - abs(p_az - t_az))
            el_error = abs(p_cola - t_cola)
            dist = np.sqrt(az_error ** 2 + el_error ** 2)
            if dist < min_dist and dist <= max_error: # 最小化误差
                min_dist = dist
                best_idx = p_idx
        if best_idx != -1:
            matched.append((t_az, t_cola, pred_angles[best_idx][0], pred_angles[best_idx][1]))
            used_pred.add(best_idx)
            if min_dist <= max_error:
                correct_matches += 1
        else:
            matched.append((t_az, t_cola, None, None))

    # 添加虚警预测
    for p_idx, (p_az, p_cola) in enumerate(pred_angles):
        if p_idx not in used_pred:
            matched.append((None, None, p_az, p_cola))

    # 统计漏检和错检数量
    missed_detections = len([m for m in matched if m[2] is None])
    false_alarms = len([m for m in matched if m[0] is None])

    return matched, missed_detections, false_alarms, correct_matches


def main(folder_path="/home/zengkehan/voice/multisource_dataset"):
    true_cola_all = []
    true_az_all = []
    pred_az_all = []
    pred_cola_all = []
    total_missed_detections = 0
    total_false_alarms = 0
    total_correct_matches = 0
    total_true_sources = 0

    # 设置参数
    room_dimension = np.array([50, 50, 50])  # 房间尺寸
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
    colatitude_search = np.radians(np.linspace(27, 90, 64))

    base_folder = folder_path
    for i in range(22, 100):
        folder = os.path.join(base_folder, "wavs", f"sample_{i}")
        try:
            freqsignals, fs = load_wav_files_to_freqdomain(folder, 64)
        except:
            continue

        # 读取真实标签
        label_file = os.path.join(base_folder, "metadata", f"sample_{i}.json")
        with open(label_file, 'r') as f:
            metadata = json.load(f)

        # 提取多声源信息
        true_sources = []
        for src in metadata.get('sources', []):
            az = np.radians(src['azimuth_deg'])
            el = np.radians(src['elevation_deg'])
            true_sources.append((np.degrees(az), np.degrees(np.pi / 2 - el)))  # (azimuth, colatitude)

        # 动态设置检测声源数量
        num_src = len(true_sources)
        total_true_sources += num_src

        # 进行DOA估计
        doa = pra.doa.srp.SRP(mic_positions, fs, nfft=1024,
                              azimuth=azimuth_search, colatitude=colatitude_search,
                              dim=3, freq_range=[100, 6000], num_src=max(1, num_src)) # 至少1个声源
        try:
            doa.locate_sources(freqsignals)
            # 获取估计结果
            pred_az = np.degrees(doa.azimuth_recon)
            pred_cola = np.degrees(doa.colatitude_recon)
            pred_sources = list(zip(pred_az, pred_cola))
        except ValueError:
            print(f"样本 {i} 检测失败（可能无足够声源）")
            pred_sources = []

        # 匹配声源
        matches, missed_detections, false_alarms, correct_matches = match_sources(true_sources, pred_sources)
        total_missed_detections += missed_detections
        total_false_alarms += false_alarms
        total_correct_matches += correct_matches

        # 收集匹配结果
        for t_az, t_cola, p_az, p_cola in matches:
            if t_az is not None and p_az is not None:
                true_az_all.append(np.radians(t_az))
                true_cola_all.append(np.radians(t_cola))
                pred_az_all.append(np.radians(p_az))
                pred_cola_all.append(np.radians(p_cola))
            # elif t_az is not None:  # 漏检
            #     true_az_all.append(np.radians(t_az))
            #     true_cola_all.append(np.radians(t_cola))
            #     pred_az_all.append(np.radians(t_az))  # 使用真实值占位
            #     pred_cola_all.append(np.radians(t_cola))
            # elif p_az is not None:  # 虚警
            #     true_az_all.append(np.radians(p_az))  # 使用预测值占位
            #     true_cola_all.append(np.radians(p_cola))
            #     pred_az_all.append(np.radians(p_az))
            #     pred_cola_all.append(np.radians(p_cola))

        # 打印匹配结果
        logger.info(f"样本 {i} 真实声源: {true_sources}；预测声源: {pred_sources}")

    # 输出统计结果
    # print(f"总真实声源数量: {total_true_sources}")
    # print(f"总漏检数量: {total_missed_detections}")
    # print(f"总错检数量: {total_false_alarms}")
    # print(f"总定位正确数量: {total_correct_matches}")
    logger.info(f"总真实声源数量: {total_true_sources}")
    logger.info(f"总漏检数量: {total_missed_detections}")
    logger.info(f"总错检数量: {total_false_alarms}")
    logger.info(f"总定位正确数量: {total_correct_matches}")

    # 可视化
    true_elevation_all = np.pi / 2 - np.array(true_cola_all)
    pred_elevation_all = np.pi / 2 - np.array(pred_cola_all)
    ame, cme = plot_joint_error_heatmap(true_az_all, true_elevation_all, pred_az_all, pred_elevation_all)
    # print(f"平均方位角误差: {ame}°")
    # print(f"平均余纬度误差: {cme}°")
    logger.info(f"平均方位角误差: {ame}°")
    logger.info(f"平均余纬度误差: {cme}°")

if __name__ == '__main__':
    main("../voice/multisource4eval_3")