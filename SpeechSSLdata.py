#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/5 14:20
# @Author : 箴澄
# @Func : 使用Pyroomacoustics库生成GoogleSpeechCommand声源定位的仿真数据 1s,16kHz,远场,归一化到[-255,255], 噪声可选
# @File : SpeechSSLdata.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
# from pyroomacoustics.datasets import GoogleSpeechCommands
import random
import soundfile as sf
import json
import os
import pandas as pd
# from scipy.io import wavfile


def generate_sound_source_localization_dataset(
        num_samples=1000,  # 生成的样本数量
        room_dimension=(50, 50, 50),  # 房间尺寸
        mic_array_height=0.0,  # 麦克风阵列高度（置于地面）
        mic_length=0.2,  # 麦克风阵列边长
        mic_num_per_line=8  # 麦克风阵列每行麦克风数量
):
    """
    生成单声源直达声定位数据集
    """
    # 配置基础路径
    # signal_folder = "/home/zengkehan/voice/bal_train"
    # output_base = "/home/zengkehan/voice/single64_dataset"
    signal_folder = "/home/zengkehan/voice/google_speech_commands"
    output_base = "/home/zengkehan/voice/speech_snr_30"

    # 创建输出目录
    os.makedirs(os.path.join(output_base, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "metadata"), exist_ok=True)

    # 麦克风阵列位置
    mic_positions = np.zeros((3, mic_num_per_line * mic_num_per_line))
    spacing = mic_length / (mic_num_per_line - 1)  # 麦克风间距
    offset = (mic_num_per_line - 1) / 2  # 中心对称偏移量
    for i in range(mic_num_per_line):
        for j in range(mic_num_per_line):
            index = i * mic_num_per_line + j
            mic_positions[0, index] = room_dimension[0] / 2 + (i - offset) * spacing
            mic_positions[1, index] = room_dimension[1] / 2 + (j - offset) * spacing
            mic_positions[2, index] = mic_array_height

    # 加载声源库
    speech_signal = pra.datasets.GoogleSpeechCommands(basedir=signal_folder, download=True, subset=1, seed=0)
    # flac_files = [f for f in os.listdir(signal_folder) if f.endswith(".flac")]
    classes = speech_signal.classes

    sample_idx = 0
    while sample_idx < num_samples:
        # 随机选择一个类别中的一个样本
        random_class = random.choice(classes)
        print(f"随机选择的语音类别: {random_class}")
        all_samples = speech_signal.filter(word=random_class)
        if not all_samples:
            print(f"类别 {random_class} 中没有样本")
            continue
        # 随机选择一个样本作为声源
        random_sample = random.choice(all_samples)
        fs = random_sample.fs
        # 获取该样本的文件路径
        audio_filepath = random_sample.meta.file_loc

        # 声源位置随机生成
        r = np.random.uniform(5, 20) # 只考虑远离麦克风阵列的声源
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi / 2)

        source_x = r * np.cos(phi) * np.cos(theta) + room_dimension[0] / 2
        source_y = r * np.cos(phi) * np.sin(theta) + room_dimension[1] / 2
        source_z = r * np.sin(phi)
        source_position = np.array([source_x, source_y, source_z])

        # 创建空房间
        room = pra.ShoeBox(
            room_dimension,
            fs=fs,
            absorption=1.0,  # 完全吸收
            max_order=0  # 无反射
        )
        # 添加麦克风阵列
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        # 添加声源
        room.add_source(source_position)
        # fig, ax = room.plot(mic_marker_size=30)
        # ax.set_xlim([0, 50])
        # ax.set_ylim([0, 50])
        # ax.set_zlim([0, 50])
        # plt.show()
        # source_signal = audio / np.max(np.abs(audio))  # 归一化输入信号
        room.sources[0].signal = random_sample.data.astype(np.float32)

        # 生成多通道录音
        room.simulate()
        multichannel_signal = room.mic_array.signals

        # 保存 WAV 文件
        room_dir = os.path.join(output_base, "wavs", f"sample_{sample_idx}")
        for i in range(multichannel_signal.shape[0]):
            os.makedirs(room_dir, exist_ok=True)
            wav_filename = f'{room_dir}/channel_{i}.wav'
            # 归一化到[-255, 255]
            multi_audio255 = multichannel_signal[i] / np.max(np.abs(multichannel_signal[i])) * 255
            # 添加高斯噪声
            multi_audio255 = add_gaussian_noise(multi_audio255, snr=30)
            sf.write(wav_filename, multi_audio255, fs)

        # 生成并保存 JSON 元数据
        metadata = {
            'source_file': audio_filepath,
            'source_position': source_position.tolist(),
            'source_distance': r,
            'source_azimuth': theta,
            'source_elevation': phi,
        }
        json_filename = os.path.join(output_base, "metadata", f"sample_{sample_idx}.json")
        with open(json_filename, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"已生成第 {sample_idx+1} 个样本")
        sample_idx += 1


def add_gaussian_noise(signal, snr):
    """
    添加高斯白噪声

    参数:
    - signal: 原始信号
    - snr: 信噪比
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


if __name__ == '__main__':
    # np.random.seed(42)
    generate_sound_source_localization_dataset()
