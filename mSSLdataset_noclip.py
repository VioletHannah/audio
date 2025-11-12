#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/11/11 11:39
# @Author : 箴澄
# @Func：
# @File : mSSLdataset_noclip.py
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/5 12:55
# @Author : 箴澄
# @File : mSSLdataset_noclip.py
# @Software: PyCharm

from util import calculate_source_intensity_welch_spectrum
import dataset_config as config
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import resample, butter, filtfilt, sosfiltfilt, stft
import warnings
import random
import json
import os
import glob


def generate_source_position(room_dimension, r_min=4, r_max=30, azimuth=None, elevation=None, cone_plus: bool = False):
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

    final_elevation = elevation if np.rad2deg(elevation) >= 27 else np.deg2rad(27)  # 限制最小俯仰角标签为27度
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


def analyze_source_over_time(source_signal, source_position, mic_positions, fs, window_size, hop_size):
    """
    遍历长信号，在每个窗口上计算 10 频带强度和 Welch 频谱。
    """
    if len(source_signal) < window_size:
        num_windows = 1
        source_signal = np.pad(source_signal, (0, window_size - len(source_signal)), 'constant')
    else:
        num_windows = 1 + (len(source_signal) - window_size) // hop_size
        if (len(source_signal) - window_size) % hop_size != 0:
            num_windows += 1

    all_welch_spectrums = []
    freqs_welch_out = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size

        if end > len(source_signal):
            start = max(0, len(source_signal) - window_size)
            end = len(source_signal)

        segment = source_signal[start:end]

        if len(segment) < window_size:
            segment = np.pad(segment, (0, window_size - len(segment)), 'constant')

        try:
            spectrum, freqs = calculate_source_intensity_welch_spectrum(
                segment,
                source_position,
                mic_positions,
                fs=fs,
                freq_bins=config.WELCH_FREQ_BINS,
                nperseg=config.WELCH_NPERSEG
            )
            all_welch_spectrums.append(list(spectrum))
            if i == 0:
                freqs_welch_out = list(freqs)
        except Exception as e:
            warnings.warn(f"Welch 频谱计算失败 (窗口 {i}): {e}")
            all_welch_spectrums.append([0.0] * config.WELCH_FREQ_BINS)

        if end == len(source_signal):
            break

    return all_welch_spectrums, freqs_welch_out

def generate_multi_source_dataset(
        num_samples=1000,
        room_dimension=(120, 120, 100),
        mic_length=0.12,  # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        dataset_path="path/to/nsynth-train/audio",
        output_path="/home/zengkehan/voice/multisource_with_intensity",
        max_sources=3,
        fs=config.FS
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

    # 记录每个样本的功率信息
    for sample_idx in range(num_samples):
        metadata = {
            'sources': [],
            'source_files': [],
            'num_sources': 0,
            'source_positions': [],
            'mic_positions': mic_positions.tolist(),
            'welch_spectrums_over_time': [],
            'welch_freqs': []
        }
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

            def _prepare_audio_segment(audio, orig_fs, target_fs):
                audio = np.asarray(audio)
                if audio.size == 0:
                    # 返回空数组，而不是固定长度的零
                    return np.zeros(0, dtype=np.float32)
                # 转为单声道
                if audio.ndim > 1:
                    if audio.shape[1] == 2:
                        audio = np.mean(audio, axis=1)
                    else:
                        audio = audio[:, 0]
                # 统一采样率
                if orig_fs != target_fs and len(audio) > 0:
                    new_len = int(np.round(len(audio) * float(target_fs) / float(orig_fs)))
                    if new_len <= 0:
                        return np.zeros(0, dtype=np.float32)
                    audio = resample(audio, new_len)

                audio = np.asarray(audio, dtype=np.float32)
                return audio

            audio = _prepare_audio_segment(audio, orig_fs, fs)

            if audio.size == 0:
                print(f"  - 警告：文件 {os.path.basename(audio_path)} 处理后为空，跳过。")
                continue

            # 对声源信号进行缩放
            from util import rms_scaling
            audio = rms_scaling(audio)

            # 创建声源
            position, azimuth, elevation = generate_source_position(room_dimension, r_min=4, r_max=5)
            # room.add_source(position)
            # room.sources[-1].signal = audio
            source_signals.append(audio)
            source_positions.append(position)

            # 保存元数据 - 方位角和俯仰角
            metadata['sources'].append({
                'azimuth_deg': float(np.rad2deg(azimuth)),
                'elevation_deg': float(np.rad2deg(elevation)),
            })
            metadata['source_files'].append(os.path.basename(audio_path))

        num_sources = len(source_signals)
        metadata['num_sources'] = num_sources  # 在元数据中明确记录最终的声源数

        # 在模拟前，将所有源信号填充到最大长度
        if num_sources > 0:
            max_len = max([len(s) for s in source_signals])
            # 确保最小长度至少为一个窗口
            max_len = max(max_len, config.WINDOW_SIZE)

            for i in range(num_sources):
                audio = source_signals[i]
                if len(audio) < max_len:
                    source_signals[i] = np.pad(audio, (0, max_len - len(audio)), 'constant')
                room.add_source(source_positions[i], signal=source_signals[i])

        # 计算每个声源的实际信号强度
        welch_spectrums_list_of_lists = []
        freqs_welch = []

        for i in range(num_sources):
            spectrum_list, freqs = analyze_source_over_time(
                source_signals[i],
                source_positions[i],
                mic_positions,
                fs=fs,
                window_size=config.WINDOW_SIZE,
                hop_size=config.HOP_SIZE
            )
            welch_spectrums_list_of_lists.append(spectrum_list)

            if i == 0:
                freqs_welch = freqs

        metadata['welch_spectrums_over_time'] = welch_spectrums_list_of_lists
        metadata['welch_freqs'] = freqs_welch

        # 模拟房间声学，获得多通道信号
        if num_sources == 0:
            # 无声源时，生成1秒的静音
            silence_len = max(fs, config.WINDOW_SIZE)  # 确保至少为一个窗口长度
            mixed_signal = np.zeros((mic_positions.shape[1], silence_len), dtype=np.float64)
        else:
            # 有声源时进行房间声学模拟, pyroomacoustics 会自动将混合信号的长度设置为最长源的长度
            room.simulate()
            mixed_signal = room.mic_array.signals

        # 保存多通道音频和标签元数据
        room_audio_dir = os.path.join(output_base, "wavs", f"sample_{sample_idx}")
        os.makedirs(room_audio_dir, exist_ok=True)
        output_wav_path = os.path.join(room_audio_dir, "mix.wav")
        sf.write(output_wav_path, mixed_signal.T, fs)

        with open(os.path.join(output_base, "metadata", f"sample_{sample_idx}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        # 打印的日志现在包含信号的实际长度
        print(f"生成样本 {sample_idx + 1}/{num_samples}，包含 {num_sources} 个声源，长度: {mixed_signal.shape[1]} 采样点")


if __name__ == '__main__':
    generate_multi_source_dataset(
        dataset_path="/home/kehan.zeng/DATA2/librispeech/LibriSpeech/test-clean",
        output_path="/home/kehan.zeng/DATA2/voice/audioset_noclip",
        num_samples=10000,
        room_dimension=(120, 120, 100),
        mic_length=0.12,  # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        max_sources=3
    )