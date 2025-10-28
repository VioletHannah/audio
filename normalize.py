#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/13 20:59
# @Author : 箴澄
# @Func：计算
# @File : normalize.py
# @Software: PyCharm
import json
from sys import meta_path

import numpy as np
import soundfile as sf
import os
import shutil

from sklearn.cluster import mean_shift
from tqdm import tqdm

def escalate_intensity(dataset, output):
    """
    对数据集进行强度提升处理
    """
    k = np.load(os.path.join(dataset, "global_std.npy"))
    meta_dir = os.path.join(dataset, "metadata")

    # 修改元数据中的强度值
    metadata_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    for file in metadata_files:
        file_path = os.path.join(meta_dir, file)
        with open(file_path, 'r') as f:
            metadata = json.load(f)

        # 提升强度
        for sample in metadata['samples']:
            sample['intensity'] /= k

        # 保存修改后的元数据
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

def normalize_dataset(dataset_path, output_path):
    """
    将数据集归一化处理
    :param dataset_path: 原始数据集路径
    :param output_path: 归一化后的数据集输出路径
    """
    # 创建输出目录结构
    os.makedirs(os.path.join(output_path, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "metadata"), exist_ok=True)

    # 加载全局统计量
    global_mean = np.load(os.path.join(dataset_path, "global_mean.npy"))
    global_std = np.load(os.path.join(dataset_path, "global_std.npy"))
    print(f"加载全局统计量: 均值 = {global_mean:.6f}, 标准差 = {global_std:.6f}")

    # 复制元数据文件
    metadata_src = os.path.join(dataset_path, "metadata")
    metadata_dst = os.path.join(output_path, "metadata")
    if os.path.exists(metadata_dst):
        shutil.rmtree(metadata_dst)
    shutil.copytree(metadata_src, metadata_dst)
    print("元数据文件复制完成")

    # 复制全局统计文件
    shutil.copy(os.path.join(dataset_path, "global_mean.npy"), output_path)
    shutil.copy(os.path.join(dataset_path, "global_std.npy"), output_path)

    # 获取所有样本目录
    wavs_dir = os.path.join(dataset_path, "wavs")
    samples = [d for d in os.listdir(wavs_dir) if os.path.isdir(os.path.join(wavs_dir, d))]

    print(f"开始处理 {len(samples)} 个样本...")
    for sample_dir in tqdm(samples):
        src_sample_path = os.path.join(wavs_dir, sample_dir)
        dst_sample_path = os.path.join(output_path, "wavs", sample_dir)
        os.makedirs(dst_sample_path, exist_ok=True)

        # 处理每个通道的音频文件
        for ch_file in os.listdir(src_sample_path):
            if ch_file.endswith(".wav"):
                src_file = os.path.join(src_sample_path, ch_file)
                dst_file = os.path.join(dst_sample_path, ch_file)

                # 读取音频数据
                data, fs = sf.read(src_file, dtype='float64')

                # 归一化处理
                normalized_data = (data - global_mean) / global_std

                # 保存归一化后的音频
                sf.write(dst_file, normalized_data, fs)

    print("数据集归一化完成！")


def normalize_intensity(original_path, dataset_path):
    # meta_dst = os.path.join(dataset_path, "metadata")
    # origin_meta_data = os.path.join(original_path, "metadata")
    # for file in os.listdir(origin_meta_data):
    #     if file.endswith('.json'):
    #         src_file = os.path.join(origin_meta_data, file)
    #         dst_file = os.path.join(meta_dst, file)
    #
    #         # 读取源文件内容
    #         with open(src_file, 'r') as f:
    #             src_metadata = json.load(f)
    #
    #         # 如果目标文件存在，读取并更新；否则直接复制
    #         if os.path.exists(dst_file):
    #             with open(dst_file, 'r') as f:
    #                 dst_metadata = json.load(f)
    #             # 将源文件的值复制到目标文件中
    #             dst_metadata.update(src_metadata)
    #         else:
    #             dst_metadata = src_metadata
    #
    #         # 保存更新后的文件
    #         with open(dst_file, 'w') as f:
    #             json.dump(dst_metadata, f, indent=4)

    meta_dir = os.path.join(dataset_path, "metadata")
    metadata_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    # mean_intensity = 0.184789
    # std_intensity = 0.902371
    # 归一化强度值
    for file in metadata_files:
        file_path = os.path.join(meta_dir, file)
        with open(file_path, 'r') as f:
            metadata = json.load(f)

        for sample in metadata.get('intensities', []):
            for i in range(len(sample.get('intensities', []))):
                inten = sample['intensities'][i]
                sample['intensities'][i] = 20 * np.log10(inten / 20e-6) if inten > 20e-6 else 0

        # 保存修改后的元数据
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    print("强度归一化处理完成！")

def statistics_dataset(dataset_path):
    """
    查看数据集的每个样本
    :param dataset_path: 数据集路径
    """
    wavs_dir = os.path.join(dataset_path, "wavs")
    samples = [d for d in os.listdir(wavs_dir) if os.path.isdir(os.path.join(wavs_dir, d))]

    for sample_dir in tqdm(samples):
        sample_path = os.path.join(wavs_dir, sample_dir)
        # 选择第一个通道
        file_path = os.path.join(sample_path, "channel_0.wav")
        data, _ = sf.read(file_path, dtype='float64')
        print(f"{file_path}: 均值={np.mean(data):.6f}, 最大值={np.max(data):.6f}, 最小值={np.min(data):.6f}")

    meta_dir = os.path.join(dataset_path, "metadata")
    metadata_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    for file in metadata_files:
        file_path = os.path.join(meta_dir, file)
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        print(f"{file_path}: 声源数量={len(metadata.get('sources_files', []))}")
        for sample in metadata.get('intensities', []):
            for intensity in sample.get('intensities', []):
                print(f"  强度值={intensity:.6f}")
    print("数据集统计完成！")


if __name__ == "__main__":
    # 配置路径
    original_dataset = "/home/kehan.zeng/DATA2/voice/multisource_with_intensity"
    normalized_dataset = "/home/kehan.zeng/DATA2/voice/multisource_normalized"

    normalize_intensity(original_dataset, normalized_dataset)

    # statistics_dataset(normalized_dataset)
    # normalize_dataset(original_dataset, normalized_dataset)