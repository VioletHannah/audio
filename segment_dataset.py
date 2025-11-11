#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/6 10:00
# @Author : 箴澄
# @File : segment_dataset.py
# @Software: PyCharm

import os
import glob
import json
import numpy as np
import soundfile as sf
import warnings
from tqdm import tqdm


def create_segmented_dataset(noclip_dataset_path,
                             segmented_dataset_path,
                             window_size=16000,
                             overlap=0.5):
    """
    将 mSSLdataset_noclip.py 生成的长音频文件切割成有重叠的短片段，
    并保存为 load_data.py 期望的格式 (每个通道一个文件)。

    参数:
    noclip_dataset_path (str): 'noclip' 数据集的根目录 (包含 'wavs' 和 'metadata')
    segmented_dataset_path (str): 切片后新数据集的保存根目录
    window_size (int): 每个训练片段的长度 (例如 16000 采样点)
    overlap (float): 窗口重叠率 (例如 0.5 表示 50% 重叠)
    """

    print(f"开始处理 'noclip' 数据集: {noclip_dataset_path}")
    print(f"将要创建的分段数据集: {segmented_dataset_path}")
    print(f"窗口大小: {window_size}, 重叠: {overlap * 100}%")

    # (中文注释) 1. 定义输入路径
    noclip_wav_dir = os.path.join(noclip_dataset_path, "wavs")
    noclip_meta_dir = os.path.join(noclip_dataset_path, "metadata")

    # (中文注释) 2. 创建输出路径
    output_wav_dir = os.path.join(segmented_dataset_path, "wavs")
    output_meta_dir = os.path.join(segmented_dataset_path, "metadata")
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(output_meta_dir, exist_ok=True)

    # (中文注释) 3. 计算步长 (Hop Size)
    hop_size = int(window_size * (1 - overlap))
    if hop_size <= 0:
        raise ValueError("Hop size 必须大于 0。请检查 window_size 和 overlap 设置。")

    # (中文注释) 4. 查找所有的 'noclip' 元数据
    # 我们以元数据为准，查找对应的音频
    noclip_meta_files = sorted(glob.glob(os.path.join(noclip_meta_dir, "sample_*.json")))

    if not noclip_meta_files:
        print(f"警告：在 {noclip_meta_dir} 中未找到 'sample_*.json' 文件。")
        return

    global_segment_index = 0  # (中文注释) 全局的、唯一的切片ID

    # (中文注释) 5. 遍历每一个 'noclip' 样本 (长音频)
    for meta_path in tqdm(noclip_meta_files, desc="处理长音频文件"):
        try:
            # (中文注释) 5.1. 加载元数据
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # (中文注释) 5.2. 构造对应的 mix.wav 路径
            sample_name = os.path.basename(meta_path).replace('.json', '')  # e.g., "sample_0"
            audio_file_path = os.path.join(noclip_wav_dir, sample_name, "mix.wav")

            if not os.path.exists(audio_file_path):
                warnings.warn(f"跳过：找不到对应的音频文件 {audio_file_path}")
                continue

            # (中文注释) 5.3. 加载长音频 mix.wav
            # soundfile.read 加载格式为 (samples, channels)
            audio_data, fs = sf.read(audio_file_path, dtype='float32')

            if audio_data.ndim == 1:
                # (中文注释) 容错处理，万一保存的是单通道
                audio_data = audio_data.reshape(-1, 1)

            total_samples = audio_data.shape[0]
            num_channels = audio_data.shape[1]

            if total_samples < window_size:
                # (中文注释) 5.4. 如果音频太短，进行零填充 (Pad)
                warnings.warn(f"文件 {audio_file_path} 长度 ({total_samples}) 小于窗口 ({window_size})，将进行零填充。")
                pad_width = window_size - total_samples
                segment_data = np.pad(audio_data, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
                start_indices = [0]  # 只有一个切片
            else:
                # (中文注释) 5.5. 如果音频够长，计算所有切片的起始点
                start_indices = list(range(0, total_samples - window_size + 1, hop_size))
                # (中文注释) 确保最后一个可能的窗口被包括 (如果它不完全重叠)
                if (total_samples - window_size) % hop_size != 0:
                    last_start = total_samples - window_size
                    if last_start > start_indices[-1]:
                        start_indices.append(last_start)

            # (中文注释) 5.6. 遍历该文件的所有切片
            for start in start_indices:
                end = start + window_size
                segment_data = audio_data[start:end, :]  # (window_size, num_channels)

                # (中文注释) 5.7. 创建新的切片目录 (e.g., .../wavs/segment_101)
                new_segment_name = f"segment_{global_segment_index}"
                new_segment_wav_dir = os.path.join(output_wav_dir, new_segment_name)
                os.makedirs(new_segment_wav_dir, exist_ok=True)

                # (中文注释) 5.8. 【关键】分离通道并保存
                # 这正是为了适配你的 load_data.py
                for ch_idx in range(num_channels):
                    channel_data = segment_data[:, ch_idx]  # (window_size,)
                    channel_output_path = os.path.join(new_segment_wav_dir, f"channel_{ch_idx}.wav")
                    sf.write(channel_output_path, channel_data, fs)

                # (中文注释) 5.9. 复制元数据
                # 所有的切片共享来自 'noclip' 样本的相同元数据 (DOA, 强度等)
                new_segment_meta_path = os.path.join(output_meta_dir, f"{new_segment_name}.json")
                with open(new_segment_meta_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                global_segment_index += 1  # (中文注释) 更新全局ID

        except Exception as e:
            warnings.warn(f"处理 {meta_path} 时发生错误: {e}")

    print("\n-------------------------------------------------")
    print("分段数据集创建完成！")
    print(f"总共处理了 {len(noclip_meta_files)} 个长音频文件。")
    print(f"共生成了 {global_segment_index} 个训练切片。")
    print(f"数据保存在: {segmented_dataset_path}")
    print("-------------------------------------------------")


if __name__ == '__main__':
    # (中文注释) --- 配置路径 ---

    # (中文注释) 1. 'noclip' 数据集的路径 (mSSLdataset_noclip.py 的输出)
    NOCLIP_DATA_PATH = "/home/kehan.zeng/DATA2/voice/mssl_libri_cone_noclip"

    # (中文注释) 2. 新的、分段后的数据集的保存路径
    SEGMENTED_DATA_PATH = "/home/kehan.zeng/DATA2/voice/mssl_libri_cone_segmented"

    # (中文注释) 3. 切片参数
    TARGET_WINDOW_SIZE = 16000  # (中文注释) 你网络期望的输入长度
    TARGET_OVERLAP = 0.5  # (中文注释) 50% 重叠

    # (中文注释) --- 开始执行 ---
    create_segmented_dataset(
        noclip_dataset_path=NOCLIP_DATA_PATH,
        segmented_dataset_path=SEGMENTED_DATA_PATH,
        window_size=TARGET_WINDOW_SIZE,
        overlap=TARGET_OVERLAP
    )