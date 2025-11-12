#!/usr/bin/env python
# -*- coding: utf-8 -*-\
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

# from util import calculate_source_intensity_welch_spectrum

import dataset_config as config


def create_segmented_dataset(noclip_dataset_path,
                             segmented_dataset_path,
                             window_size,  # (中文注释) 修改：从 config 传入
                             overlap  # (中文注释) 修改：从 config 传入
                             ):
    """
    将 mSSLdataset_noclip.py 生成的长音频文件切割成有重叠的短片段

    参数:
    noclip_dataset_path (str): 'noclip' 数据集的根目录 (包含 'wavs' 和 'metadata')
    segmented_dataset_path (str): 切片后新数据集的保存根目录
    window_size (int): 每个训练片段的长度 (例如 16000 采样点)
    overlap (float): 窗口重叠率 (例如 0.5 表示 50% 重叠)
    """

    print(f"开始处理 'noclip' 数据集: {noclip_dataset_path}")
    print(f"将要创建的分段数据集: {segmented_dataset_path}")
    print(f"窗口大小: {window_size}, 重叠: {overlap * 100}%")

    noclip_wav_dir = os.path.join(noclip_dataset_path, "wavs")
    noclip_meta_dir = os.path.join(noclip_dataset_path, "metadata")

    output_wav_dir = os.path.join(segmented_dataset_path, "wavs")
    output_meta_dir = os.path.join(segmented_dataset_path, "metadata")
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(output_meta_dir, exist_ok=True)

    hop_size = int(window_size * (1 - overlap))
    if hop_size <= 0:
        raise ValueError("Hop size 必须大于 0。请检查 window_size 和 overlap。")

    noclip_meta_files = sorted(glob.glob(os.path.join(noclip_meta_dir, "sample_*.json")))

    if not noclip_meta_files:
        print(f"警告：在 {noclip_meta_dir} 中未找到 'sample_*.json' 文件。")
        return

    total_segments_created = 0

    for meta_path in tqdm(noclip_meta_files, desc="处理长音频文件"):
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            sample_name = os.path.basename(meta_path).replace('.json', '')  # e.g., "sample_0"
            audio_file_path = os.path.join(noclip_wav_dir, sample_name, "mix.wav")

            if not os.path.exists(audio_file_path):
                warnings.warn(f"跳过：找不到对应的音频文件 {audio_file_path}")
                continue

            audio_data, fs = sf.read(audio_file_path)
            if fs != config.FS:
                warnings.warn(f"文件 {audio_file_path} 的采样率 {fs} 与配置 {config.FS} 不符。")

            if audio_data.ndim == 1:
                audio_data = np.expand_dims(audio_data, axis=1)

            total_samples = audio_data.shape[0]
            num_channels = audio_data.shape[1]

            output_wav_dir_for_sample = os.path.join(output_wav_dir, sample_name)
            output_meta_dir_for_sample = os.path.join(output_meta_dir, sample_name)
            os.makedirs(output_wav_dir_for_sample, exist_ok=True)
            os.makedirs(output_meta_dir_for_sample, exist_ok=True)

            if total_samples < window_size:
                pad_width = window_size - total_samples
                audio_data = np.pad(audio_data, ((0, pad_width), (0, 0)), 'constant')
                total_samples = window_size
                start_indices = [0]
            else:
                start_indices = list(range(0, total_samples - window_size + 1, hop_size))
                if (total_samples - window_size) % hop_size != 0:
                    last_start = total_samples - window_size
                    if last_start > start_indices[-1]:
                        start_indices.append(last_start)

            num_sources = metadata.get('num_sources', 0)
            num_segments_in_metadata = 0
            if num_sources > 0 and metadata.get('welch_spectrums_over_time'):
                try:
                    num_segments_in_metadata = len(metadata['welch_spectrums_over_time'][0])
                except IndexError:
                    num_segments_in_metadata = 0  # 可能是空的声源列表

            # 混合信号和分析列表的长度应一致
            num_segments_to_create = min(len(start_indices), num_segments_in_metadata)

            # 处理静音文件
            if num_sources == 0:
                num_segments_to_create = len(start_indices)

            # 遍历所有有效的切片
            for local_segment_index in range(num_segments_to_create):

                start = start_indices[local_segment_index]
                end = start + window_size
                segment_data = audio_data[start:end, :]  # (window_size, num_channels)

                new_segment_name = f"segment_{local_segment_index}"
                new_segment_wav_dir = os.path.join(output_wav_dir_for_sample, new_segment_name)
                os.makedirs(new_segment_wav_dir, exist_ok=True)

                for ch_idx in range(num_channels):
                    channel_data = segment_data[:, ch_idx]  # (window_size,)
                    channel_output_path = os.path.join(new_segment_wav_dir, f"channel_{ch_idx}.wav")
                    sf.write(channel_output_path, channel_data, fs)

                segment_metadata = {
                    'sources': metadata['sources'],
                    'num_sources': num_sources,
                    'welch_freqs': metadata.get('welch_freqs', []),
                    'source_positions': metadata.get('source_positions', []),
                    'mic_positions': metadata.get('mic_positions', []),
                    'welch_spectrums': []
                }

                # 提取此切片的数据
                for i in range(num_sources):
                    try:
                        segment_metadata['welch_spectrums'].append(
                            metadata['welch_spectrums_over_time'][i][local_segment_index]
                        )
                    except IndexError:
                        warnings.warn(f"在 {sample_name} / segment {local_segment_index} 索引 {i} 处出错")
                        segment_metadata['welch_spectrums'].append([0.0] * config.WELCH_FREQ_BINS)

                new_segment_meta_path = os.path.join(output_meta_dir_for_sample, f"{new_segment_name}.json")
                with open(new_segment_meta_path, 'w') as f:
                    json.dump(segment_metadata, f, indent=4)

                total_segments_created += 1

        except Exception as e:
            warnings.warn(f"处理 {meta_path} 时发生严重错误: {e}")

    print("\n-------------------------------------------------")
    print("分段数据集创建完成！")
    print(f"总共处理了 {len(noclip_meta_files)} 个长音频文件。")
    print(f"共生成了 {total_segments_created} 个训练切片。")
    print(f"数据保存在: {segmented_dataset_path}")
    print("-------------------------------------------------")


if __name__ == '__main__':
    NOCLIP_DATA_PATH = "/home/kehan.zeng/DATA2/voice/mssl_libri_cone_noclip"
    SEGMENTED_DATA_PATH = "/home/kehan.zeng/DATA2/voice/mssl_libri_cone_segmented"

    TARGET_WINDOW_SIZE = config.WINDOW_SIZE
    TARGET_OVERLAP = config.OVERLAP

    create_segmented_dataset(
        NOCLIP_DATA_PATH,
        SEGMENTED_DATA_PATH,
        window_size=TARGET_WINDOW_SIZE,
        overlap=TARGET_OVERLAP
    )