#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/23 15:07
# @Author : 箴澄
# @Func : 生成基于 NSynth 数据集的多声源定位数据集
# @File : NSDmultiSSLdata.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import random
import soundfile as sf
import json
import os
import glob
from scipy.signal import resample

def generate_source_position(room_dimension):
    """
    随机生成一个声源位置
    :param room_dimension: 房间尺寸
    :return: 声源位置数组
    """
    azimuth = np.deg2rad(np.random.uniform(0, 360))  # 方位角
    elevation = np.deg2rad(np.random.uniform(27, 91))  # 俯仰角
    r = np.random.uniform(5, 20)  # 距离

    # 转换为笛卡尔坐标
    x = r * np.cos(elevation) * np.cos(azimuth) + room_dimension[0] / 2
    y = r * np.cos(elevation) * np.sin(azimuth) + room_dimension[1] / 2
    z = r * np.sin(elevation)
    return np.array([x, y, z]), azimuth, elevation

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

def calculate_source_intensity(source_signal, source_position, mic_positions):
    """
    计算声源到各个麦克风的信号能量均值
    :param source_signal: 源信号
    :param source_position: 声源位置 [x, y, z]
    :param mic_positions: 麦克风位置数组 (3, num_mics)
    :return: 各麦克风收到的该声源信号能量均值
    """
    # 计算源信号的RMS
    source_rms = np.sqrt(np.mean(source_signal ** 2))

    # 计算每个麦克风到声源的距离
    distances = []
    for i in range(mic_positions.shape[1]):
        mic_pos = mic_positions[:, i]
        distance = np.linalg.norm(source_position - mic_pos)
        distances.append(distance)

    # 根据距离计算衰减后的能量
    received_energies = []
    for distance in distances:
        # 简单的球面波衰减模型 (1/r)
        attenuation = 1.0 / (distance + 1e-8)
        received_energy = source_rms * attenuation
        received_energies.append(received_energy)

    # 返回各麦克风收到能量的均值
    return np.mean(received_energies)

def generate_multi_source_dataset(
        num_samples=1000,
        room_dimension=(50, 50, 50),
        mic_length=0.2,
        mic_num_per_line=8,
        dataset_path="path/to/nsynth-train/audio",
        output_path="/home/zengkehan/voice/multisource_with_intensity",
        max_sources=3
):
    # 配置输出路径
    output_base = output_path
    os.makedirs(os.path.join(output_base, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "metadata"), exist_ok=True)

    # 加载NSynth音频文件列表
    audio_files = glob.glob(os.path.join(dataset_path, "*.wav"))
    if not audio_files:
        raise ValueError("未找到NSynth音频文件")

    # 创建麦克风阵列
    mic_positions = generate_mic_array_positions(mic_num_per_line, mic_length, room_dimension)

    for sample_idx in range(num_samples):
        metadata = {'sources': [], 'source_files': []}
        # 创建房间，添加麦克风阵列
        room = pra.ShoeBox(room_dimension, fs=16000, absorption=1.0, max_order=0)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, 16000))

        # 随机生成声源数量（0-3），随后添加声源信息
        # num_sources = np.random.randint(0, max_sources + 1)
        num_sources = random.choices(list(range(max_sources + 1)), [0.1, 0.2, 0.3, 0.4])[0]  # 根据权重随机选择声源数量
        for i in range(num_sources):
            # 随机选择音频文件
            audio_path = random.choice(audio_files)
            audio, fs = sf.read(audio_path)

            # 统一采样率并截取前1秒
            if fs != 16000:
                audio = resample(audio, int(16000 * len(audio) / fs))
            audio = audio[3000:19001]  # 取前1秒（16kHz）

            # 创建声源
            position, azimuth, elevation = generate_source_position(room_dimension)
            room.add_source(position)
            room.sources[i].signal = audio

            # 计算接收到该单源信号的强度
            intensity = calculate_source_intensity(audio, position, mic_positions)

            # 保存元数据
            metadata['sources'].append({
                'azimuth_deg': float(np.rad2deg(azimuth)),
                'elevation_deg': float(np.rad2deg(elevation)),
                'intensity': float(intensity),
            })
            metadata['source_files'].append(os.path.basename(audio_path))

        # 模拟房间声学，获得多通道信号

        # 预信号处理，num_sources=0时添加环境噪声只有噪声，有声源时归一化信号
        if num_sources == 0:
            noise_level = 0.01  # 无信号时的基准噪声
            mixed_signal = np.random.normal(0, noise_level, (mic_positions.shape[1], 16000))
        else:
            room.simulate()
            mixed_signal = room.mic_array.signals
            max_val = np.max(np.abs(mixed_signal))
            if max_val > 0:
                mixed_signal = mixed_signal / max_val

        # 保存多通道音频和标签元数据
        room_dir = os.path.join(output_base, "wavs", f"sample_{sample_idx}")
        os.makedirs(room_dir, exist_ok=True)
        for ch in range(mixed_signal.shape[0]):
            sf.write(f"{room_dir}/channel_{ch}.wav", mixed_signal[ch], 16000)
        with open(os.path.join(output_base, "metadata", f"sample_{sample_idx}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)


        print(f"生成样本 {sample_idx + 1}/{num_samples}，包含 {num_sources} 个声源")


if __name__ == '__main__':
    generate_multi_source_dataset(
        dataset_path="/home/zengkehan/voice/nsynth-test/audio",
        num_samples=1000
    )