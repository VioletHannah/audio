#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/21 22:21
# @Author : 箴澄
# @Func：生成不同声源数的数据集，用于测试声源数量对定位精度的影响
# @File : multiSSLdata4eval.py
# @Software: PyCharm

import numpy as np
import pyroomacoustics as pra
import random
import soundfile as sf
import json
import os
import glob
from scipy.signal import resample
from NSDmultiSSLdata import generate_source_position, generate_mic_array_positions

def generate_multi_source_dataset4eval(
        num_samples=1000,
        room_dimension=(50, 50, 50),
        mic_length=0.2,
        mic_num_per_line=8,
        num_sources=3,
        source_path="path/to/nsynth-train/audio",
        output_path="path/to/output",
):
    # 配置输出路径
    output_base = output_path
    os.makedirs(os.path.join(output_base, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "metadata"), exist_ok=True)

    # 加载NSynth音频文件列表
    audio_files = glob.glob(os.path.join(source_path, "*.wav"))
    if not audio_files:
        raise ValueError("未找到NSynth音频文件")

    # 创建麦克风阵列
    mic_positions = generate_mic_array_positions(mic_num_per_line, mic_length, room_dimension)

    for sample_idx in range(411, num_samples):
        metadata = {'sources': [], 'source_files': []}
        # 创建房间，添加麦克风阵列
        room = pra.ShoeBox(room_dimension, fs=16000, absorption=1.0, max_order=0)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, 16000))

        # 随机生成声源数量（0-3），随后添加声源信息
        # num_sources = random.choices(list(range(max_sources + 1)), [0.1, 0.2, 0.3, 0.4])[0]  # 根据权重随机选择声源数量
        for i in range(num_sources):
            # 随机选择音频文件
            audio_path = random.choice(audio_files)
            audio, fs = sf.read(audio_path)

            # 统一采样率并截取前1秒
            if fs != 16000:
                audio = resample(audio, int(16000 * len(audio) / fs))
            audio = audio[2000:18000]  # 取前1秒（16kHz）

            # 创建声源
            position, azimuth, elevation = generate_source_position(room_dimension)
            room.add_source(position)
            room.sources[i].signal = audio

            # 保存元数据
            metadata['sources'].append({
                'azimuth_deg': float(np.rad2deg(azimuth)),
                'elevation_deg': float(np.rad2deg(elevation))
            })
            metadata['source_files'].append(os.path.basename(audio_path))

        # 模拟房间声学，获得多通道信号
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
    generate_multi_source_dataset4eval(
        num_samples=500,
        source_path="/home/zengkehan/voice/nsynth-test/audio",
        output_path="/home/zengkehan/voice/multisource4eval_2",
        num_sources=2
    )