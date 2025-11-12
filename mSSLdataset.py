#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/5 12:55
# @Author : 箴澄
# @File : mSSLdataset.py
# @Software: PyCharm

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import warnings
import random
import json
import os
import glob
from util import rms_scaling, prepare_audio_segment, calculate_source_intensity_welch_spectrum


def generate_source_position(room_dimension, r_min=4, r_max=30, azimuth=None, elevation=None, cone_plus:bool=False):
    """
    随机生成一个声源位置
    :param room_dimension: 房间尺寸
    :param r_min: 最小距离
    :param r_max: 最大距离
    :param azimuth: 方位角（弧度）
    :param elevation: 俯仰角（弧度）
    :param cone_plus: 是否在锥体范围外也生成声源
    0度在x轴正方向，逆时针旋转
    俯仰角范围：27-90度（锥体内）或1-90度（锥体内外）
    0度在水平面上，90度在正上方
    :return: 声源位置数组，方位角，俯仰角（弧度）
    """
    if azimuth is None and elevation is None:
        azimuth = np.deg2rad(np.random.uniform(0, 360))  # 方位角
        if cone_plus:
            elevation = np.deg2rad(np.random.uniform(1, 91))
        else:
            elevation = np.deg2rad(np.random.uniform(27, 91))
    r = np.random.uniform(r_min, r_max)  # 距离

    # 转换为笛卡尔坐标
    x = r * np.cos(elevation) * np.cos(azimuth) + room_dimension[0] / 2
    y = r * np.cos(elevation) * np.sin(azimuth) + room_dimension[1] / 2
    z = r * np.sin(elevation)

    final_elevation = elevation if np.rad2deg(elevation) >= 27 else np.deg2rad(27) # 限制最小俯仰角标签为27度
    return np.array([x, y, z]), azimuth, final_elevation

def generate_mic_array_positions(mic_num_per_line, mic_length, room_dimension):
    """
    生成麦克风阵列的位置
    :param mic_num_per_line: 每行麦克风数量
    :param mic_length: 麦克风阵列边长
    :param room_dimension: 房间尺寸
    :return: 麦克风位置数组
    """
    mic_positions = np.zeros((3, mic_num_per_line * mic_num_per_line))
    spacing = mic_length / (mic_num_per_line - 1)
    offset = (mic_num_per_line - 1) / 2
    for i in range(mic_num_per_line):
        for j in range(mic_num_per_line):
            index = i * mic_num_per_line + j
            mic_positions[0, index] = room_dimension[0] / 2 + (i - offset) * spacing
            mic_positions[1, index] = room_dimension[1] / 2 + (j - offset) * spacing
            mic_positions[2, index] = 0
    return mic_positions

def calculate_source_intensity_over_time(source_signal, source_position, mic_positions, fs=48000, adc_range=1.0, sensitivity_mv_pa=50):
    """
    计算声源在各个时间窗口的信号强度
    :param source_signal: 源信号
    :param source_position: 声源位置 [x, y, z]
    :param mic_positions: 麦克风位置数组 (3, num_mics)
    :param fs: 采样率
    :return: 时间窗口强度列表，每个窗口的平均强度
    """
    # 计算每个麦克风到声源的距离
    distances = []
    for i in range(mic_positions.shape[1]):
        mic_pos = mic_positions[:, i]
        distance = np.linalg.norm(source_position - mic_pos)
        distances.append(distance)

    # 计算衰减因子 (1/r)
    attenuations = [1 / (d + 1e-8) for d in distances]
    avg_attenuation = np.mean(attenuations)

    # 时间窗口参数 (333ms窗口，100ms步长)
    window_length = 16000  # 样本数16000
    hop_size = int(0.1 * fs)  # 100ms的样本数
    num_windows = (len(source_signal) - window_length) // hop_size + 1

    # 计算每个时间窗口的强度
    intensities = []
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_length
        segment = source_signal[start:end]

        # 计算当前时间窗口的RMS
        p_actual = segment * (adc_range * 1000) / sensitivity_mv_pa  # 转换为 Pa
        segment_rms = np.sqrt(np.mean(p_actual ** 2))

        # 应用衰减因子
        window_intensity = segment_rms * avg_attenuation
        intensities.append(window_intensity)

    return intensities

def generate_multi_source_dataset(
        num_samples=1000,
        room_dimension=(120, 120, 100),
        mic_length=0.12, # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        dataset_path="path/to/nsynth-train/audio",
        output_path="/home/zengkehan/voice/multisource_with_intensity",
        max_sources=3,
        fs = 48000,
        samples_len = 16000,
        freq_bins=128,
        nperseg=2048
):
    # 配置输出路径
    output_base = output_path
    os.makedirs(os.path.join(output_base, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "metadata"), exist_ok=True)

    # 加载音频文件列表
    audio_files = glob.glob(os.path.join(dataset_path, "**", "*.flac"), recursive=True)
    if not audio_files:
        raise ValueError(f"未在 {dataset_path} 下找到 LibriSpeech 音频文件")

    # 创建麦克风阵列
    mic_positions = generate_mic_array_positions(mic_num_per_line, mic_length, room_dimension)

    # 滤波器组设计
    # filters, center_freqs, bandwidths = design_bandpass_filters(fs, 4)
    # filter_config = {
    #     'num_band': len(center_freqs),
    #     'center_freqs': center_freqs,
    #     'bandwidths': bandwidths,
    #     'order': 4
    # }
    # with open(os.path.join(output_base, "filter_config.json"), 'w') as f:
    #     json.dump(filter_config, f, indent=4)

    # 记录每个样本的功率信息
    for sample_idx in range(num_samples):
        metadata = {'sources': [], 'source_files': [], 'intensities': []}
        # 创建房间，添加麦克风阵列
        room = pra.ShoeBox(room_dimension, fs=fs, absorption=1.0, max_order=0)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        # 随机生成声源数量（0-3），随后添加声源信息
        num_sources = random.choices(list(range(max_sources + 1)), [0.1, 0.2, 0.3, 0.4])[0]  # 根据权重随机选择声源数量
        # 存储所有声源的原始信号和位置
        source_signals = []
        source_positions = []
        # 开始添加声源
        for i in range(num_sources):
            # 随机选择音频文件
            audio_path = random.choice(audio_files)
            audio, orig_fs = sf.read(audio_path)

            # 预处理音频片段，统一单声道，长度和采样率
            audio = prepare_audio_segment(audio, orig_fs, fs, samples_len)

            # 标准化音频功率到1
            audio = rms_scaling(audio, target_rms=0.1)

            # 创建声源
            position, azimuth, elevation = generate_source_position(room_dimension, r_min=3, r_max=10)
            room.add_source(position)
            room.sources[i].signal = audio
            source_signals.append(audio)
            source_positions.append(position)

            # 保存元数据 - 方位角和俯仰角
            metadata['sources'].append({
                'azimuth_deg': float(np.rad2deg(azimuth)),
                'elevation_deg': float(np.rad2deg(elevation)),
            })
            metadata['source_files'].append(os.path.basename(audio_path))

        # 计算每个声源的实际信号强度
        band_intensities = []
        for i in range(num_sources):
            intensities, center_freqs = calculate_source_intensity_welch_spectrum(source_signals[i], source_positions[i], mic_positions, fs, freq_bins, nperseg)
            band_intensities.append(intensities.tolist())
        metadata['intensities'] = band_intensities
        metadata['center_freqs'] = center_freqs

        # 模拟房间声学，获得多通道信号
        if num_sources == 0:
            # 无生源时生成零信号
            mixed_signal = np.zeros((mic_positions.shape[1], samples_len), dtype=np.float64)
        else:
            # 有声源时进行房间声学模拟
            room.simulate()
            mixed_signal = room.mic_array.signals

        # TODO: 噪音统一到后期添加
        # 统一添加噪声，噪声功率为信号功率的1/20
        # signal_power = np.mean(mixed_signal ** 2)
        # if signal_power < 1e-10: # 无生源时
        #     noise_power = 0.01  # 默认噪声功率
        # else:
        #     noise_power = signal_power / 20
        # # 添加噪声
        # noise = np.random.normal(0, np.sqrt(noise_power), mixed_signal.shape)
        # mixed_signal += noise

        # 保存多通道音频和标签元数据

        # 保存多通道音频和标签元数据
        room_audio_dir = os.path.join(output_base, "wavs", f"sample_{sample_idx}")
        os.makedirs(room_audio_dir, exist_ok=True)
        for ch in range(mixed_signal.shape[0]):
            sf.write(f"{room_audio_dir}/channel_{ch}.wav", mixed_signal[ch], fs)
        with open(os.path.join(output_base, "metadata", f"sample_{sample_idx}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"生成样本 {sample_idx + 1}/{num_samples}，包含 {num_sources} 个声源")

    #     # 统计样本信息
    #     sum_mean += np.mean(mixed_signal)
    #     sum_sq_mean += np.mean(mixed_signal ** 2)
    #
    # global_mean = sum_mean / num_samples / (16*16*samples_len)
    # global_std = np.sqrt(sum_sq_mean / num_samples / (16*16*samples_len) - global_mean ** 2)
    # np.save(os.path.join(output_base, f"global_mean.npy"), global_mean)
    # np.save(os.path.join(output_base, f"global_std.npy"), global_std)

if __name__ == '__main__':
    generate_multi_source_dataset(
        dataset_path="/home/kehan.zeng/DATA2/librispeech/LibriSpeech/test-clean",
        output_path="/home/kehan.zeng/DATA2/voice/mssl_libri_cone",
        num_samples=3000,
        room_dimension=(120, 120, 100),
        mic_length=0.12,  # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        max_sources=3
    )