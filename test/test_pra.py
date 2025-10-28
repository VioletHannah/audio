#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/13 14:18
# @Author : 箴澄
# @Func：测试pra模块
# @File : test_pra.py

import pyroomacoustics as pra
import soundfile as sf
import numpy as np

def test_pra():
    # 创建一个简单的房间
    room = pra.ShoeBox(
        [20, 20, 20],  # 房间尺寸
        fs=48000,     # 采样率
        absorption=1.0,  # 材料吸收系数
        max_order=0   # 最大反射次数
    )
    # room1 = pra.ShoeBox(
    #     [10, 10, 10],  # 房间尺寸
    #     fs=16000,     # 采样率
    #     max_order=10   # 最大反射次数
    # )
    # 添加一个麦克风阵列
    micpos = [[1],
              [1],
              [1]]  # 麦克风位置
    mic_array = pra.MicrophoneArray(micpos, room.fs)
    room.add_microphone_array(mic_array)
    # room1.add_microphone_array(mic_array)

    # 添加一个声源
    # audio, fs = sf.read("/home/kehan.zeng/DATA2/voice/bal_train/0NbN9By4eYw.flac")
    audio, fs = sf.read("../bal_train/y_F5Ky4cJig.flac")
    if audio.ndim > 1:
        audio = audio[:, 0]  # 如果是多通道音频，取第一个通道

    room.add_source(np.array([19,19,19]), audio)
    # room1.add_source(np.array([3,4,5]), audio)

    # 模拟房间声学
    room.simulate()
    # room1.simulate()
    # 获取麦克风接收到的信号
    mic_signal = room.mic_array.signals[0]
    # mic_signal1 = room1.mic_array.signals[0]

    # 绘制原始信号和接收到的信号
    n = 2
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(n, 1, 1)
    plt.plot(mic_signal)
    plt.title("Received Signal at Microphone")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

    # plt.subplot(n, 1, 2)
    # plt.plot(mic_signal1)
    # plt.title("Received Signal1 at Microphone")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")

    plt.subplot(n, 1, n)
    plt.plot(audio)
    plt.title("Original Audio Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # 保存接收到的信号到文件
    sf.write("received_signal_far1.wav", mic_signal, room.fs)
    sf.write("original_signal_far1.wav", audio, fs)

def count_max(folder_path="../bal_train"):
    """
    读取文件夹下每个flac文件的最大值并输出
    """
    import os
    import glob
    import soundfile as sf

    flac_files = glob.glob(os.path.join(folder_path, "*.flac"))

    max_values = []
    for file in flac_files:
        audio, fs = sf.read(file)
        if audio.ndim > 1:
            audio = audio[:, 0]  # 如果是多通道音频，取第一个通道
        max_val = np.max(np.abs(audio))
        print(f"File: {os.path.basename(file)}, Max Value: {max_val}")
        max_values.append(max_val)

    overall_max = np.mean(max_values)
    print(f"Overall maximum value across all files: {overall_max}")

count_max("/home/kehan.zeng/DATA2/voice/multisource_normalized/wavs")
# test_pra()