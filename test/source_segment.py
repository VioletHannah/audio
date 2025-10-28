#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/9/3 22:37
# @Author : 箴澄
# @Func：将音频文件分割为固定长度的音频片段
# @File : source_segment.py
# @Software: PyCharm
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse


def segment_audio_fixed_length(input_path, output_dir, segment_length=16000):
    """
    将音频文件分割为固定长度的片段

    参数:
    input_path: 输入音频文件路径
    output_dir: 输出目录
    segment_length: 每个片段的采样点数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载音频文件
        y, sr = librosa.load(input_path, sr=None)  # 保持原始采样率

        # 计算可以分割的片段数量
        num_segments = len(y) // segment_length

        if num_segments == 0:
            print(f"文件 {input_path} 太短，无法分割成 {segment_length} 个采样点的片段")
            return

        # 提取文件名（不含扩展名）
        filename = Path(input_path).stem

        # 分割音频
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = y[start:end]

            # 保存片段
            output_path = os.path.join(output_dir, f"{filename}_segment{i:04d}.flac")
            sf.write(output_path, segment, sr, format='FLAC')

        print(f"从 {input_path} 中提取了 {num_segments} 个音频片段")

    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")


def process_directory(input_dir, output_dir, segment_length=16000):
    """
    处理目录中的所有音频文件

    参数:
    input_dir: 输入目录
    output_dir: 输出目录
    segment_length: 每个片段的采样点数量
    """
    # 支持的文件格式
    supported_formats = ['.flac', '.wav', '.mp3', '.ogg']

    # 遍历目录中的所有文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                input_path = os.path.join(root, file)
                # 保持目录结构
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)

                segment_audio_fixed_length(input_path, output_subdir, segment_length)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='将音频文件分割为固定长度的片段')
    # parser.add_argument('--input_dir', type=str, required=True,
    #                     help='输入目录路径（包含音频文件）')
    # parser.add_argument('--output_dir', type=str, required=True,
    #                     help='输出目录路径')
    # parser.add_argument('--segment_length', type=int, default=16000,
    #                     help='每个片段的采样点数量，默认16000')
    #
    # args = parser.parse_args()
    #
    # process_directory(args.input_dir, args.output_dir, args.segment_length)

    process_directory("/home/kehan.zeng/DATA2/voice/bal_train", "/home/kehan.zeng/DATA2/voice/bal_train_segment")