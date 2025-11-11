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

import matplotlib.pyplot as plt
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


def calculate_source_intensity_over_time(source_signal, source_position, mic_positions, fs=48000, adc_range=1.0,
                                         sensitivity_mv_pa=50):
    """
    计算声源在各个时间窗口的信号强度
    :param source_signal: 源信号
    :param source_position: 声源位置 [x, y, z]
    :param mic_positions: 麦克风位置数组 (3, num_mics)
    :param fs: 采样率
    :return: 时间窗口强度列表，每个窗口的平均强度
    """
    # (中文注释) 修改：这里的 16000 只是一个分析窗口，与全局的 samples_len 无关，予以保留
    window_length = 16000

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
    # window_length = 16000  # 样本数16000 (已在上面定义)
    hop_size = int(0.1 * fs)  # 100ms的样本数

    # (中文注释) 新增：保护，以防信号长度小于分析窗口
    if len(source_signal) < window_length:
        return [0.0]  # 如果信号太短，返回一个默认值

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


def design_bandpass_filters(fs, order=4):
    """
    设计一组带通滤波器
    :param num_band: 频段数量
    :param fs: 采样率
    :param order: 滤波器阶数
    :return:
        滤波器系数列表 [(b1, a1), (b2, a2), ...]
        中心频率列表 [f1, f2, ...]
        带宽列表 [bw1, bw2, ...]
    """
    nyquist = fs / 2
    filters = []
    center_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    bandwidths = []

    for cf in center_freqs:
        lowcut = cf / np.sqrt(2)
        highcut = cf * np.sqrt(2)
        bandwidth = highcut - lowcut
        # print(lowcut, highcut, cf)

        lownorm = lowcut / nyquist
        highnorm = highcut / nyquist

        b, a = butter(order, [lownorm, highnorm], btype='band')

        filters.append((b, a))
        bandwidths.append(bandwidth)

    return filters, center_freqs, bandwidths


def calculate_source_intensity_by_freq(source_signal, source_position, mic_positions,
                                       fs=48000, order=4,
                                       adc_range=1.0, sensitivity_mv_pa=500):
    """
    计算声源在各频带的强度（改进版，使用sosfiltfilt保证数值稳定）
    参数：
        source_signal: ndarray, 输入信号
        source_position: (3,) 声源坐标
        mic_positions: (3, N) 麦克风坐标
        fs: 采样率
        order: 滤波器阶数
        adc_range, sensitivity_mv_pa: 电声转换参数
    返回：
        intensities: list[float], 各频带强度（近似SPL*d^-1）
        center_freqs: list[float], 中心频率
    """

    # (中文注释) 新增：如果信号太短，无法分析，直接返回0
    if len(source_signal) < 1024:  # 设置一个最小长度阈值
        warnings.warn(f"信号过短 (len={len(source_signal)})，无法计算频带强度。")
        return [0.0] * 10, [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    # 定义中心频率（ISO 1/3倍频程标准）
    center_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    # 计算平均距离衰减
    distances = np.linalg.norm(np.array(source_position).reshape(3, 1) - mic_positions, axis=0)
    avg_attenuation = np.mean(1.0 / (distances + 1e-8))

    intensities = []
    nyquist = fs / 2

    # 遍历每个频带
    for cf in center_freqs:
        lowcut = cf / np.sqrt(2)
        highcut = cf * np.sqrt(2)

        # 归一化截止频率（防止超出范围）
        low = max(lowcut / nyquist, 1e-6)
        high = min(highcut / nyquist, 0.9999)
        if low >= high:
            intensities.append(0.0)
            continue

        try:
            # 稳定的二阶节滤波器
            sos = butter(order, [low, high], btype='band', output='sos')

            # 使用 sosfiltfilt 进行零相位滤波
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filtered_signal = sosfiltfilt(sos, source_signal)

            # 数值保护：防止 inf / nan / 溢出
            filtered_signal = np.nan_to_num(filtered_signal, nan=0.0, posinf=0.0, neginf=0.0)
            filtered_signal = np.clip(filtered_signal, -1e6, 1e6)

            # 电压 -> 声压（Pa）, adc_range 单位 V, sensitivity_mv_pa 单位 mV/Pa
            p_actual = filtered_signal * (adc_range * 1000.0) / sensitivity_mv_pa

            # 计算 RMS（稳健）
            segment_rms = np.sqrt(np.mean(np.square(p_actual))) if len(p_actual) > 0 else 0.0

            # 若仍异常或过小，使用 STFT 能量估计
            if not np.isfinite(segment_rms) or segment_rms <= 1e-10:
                f, t, Zxx = stft(source_signal, fs=fs, nperseg=min(2048, len(source_signal)))
                band_mask = (f >= lowcut) & (f <= highcut)
                if np.any(band_mask):
                    band_energy = np.mean(np.abs(Zxx[band_mask, :]) ** 2)
                    segment_rms = np.sqrt(band_energy)
                else:
                    segment_rms = 0.0

            # 转换为声压级 SPL (20 μPa 参考)
            if segment_rms > 20e-6:
                spl = 20 * np.log10(segment_rms / 20e-6)
            else:
                spl = 0.0

            # 应用距离衰减
            intensity = spl * avg_attenuation
            intensities.append(float(intensity))

        except Exception as e:
            # 捕获滤波失败情况
            warnings.warn(f"Band {cf} Hz filter failed: {e}")
            intensities.append(0.0)

    return intensities, center_freqs


def calculate_source_intensity_by_freq_backup(source_signal, source_position, mic_positions, filters, fs=48000,
                                              adc_range=1.0, sensitivity_mv_pa=50):
    """
    计算声源在各个频段的信号强度
    :param source_signal: 源信号
    :param source_position: 声源位置 [x, y, z]
    :param mic_positions: 麦克风位置数组 (3, num_mics)
    :param filters: 滤波器组列表
    :param fs: 采样率
    :return: 各频段强度列表
    """
    # (中文注释) 新增：如果信号太短，无法分析，直接返回0
    if len(source_signal) < 1024:  # 示例阈值
        return [0.0] * len(filters)

    # 计算每个麦克风到声源的距离
    distances = np.linalg.norm(source_position.reshape(3, 1) - mic_positions, axis=0)
    avg_attenuation = np.mean(1 / (distances + 1e-8))

    # 初始化各频段强度
    band_intensities = []

    # 计算每个频段的强度
    for b, a in filters:
        # 滤波
        filtered_signal = filtfilt(b, a, source_signal)
        # 计算RMS
        p_actual = filtered_signal * (adc_range * 1000) / sensitivity_mv_pa  # 转换为 Pa
        segment_rms = np.sqrt(np.mean(p_actual ** 2))
        spl = 20 * np.log10(segment_rms / 20e-6) if segment_rms > 20e-6 else 0  # 转换为dB SPL
        # 应用衰减因子
        band_intensity = spl * avg_attenuation
        band_intensities.append(band_intensity)

    return band_intensities


def generate_multi_source_dataset(
        num_samples=1000,
        room_dimension=(120, 120, 100),
        mic_length=0.12,  # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        dataset_path="path/to/nsynth-train/audio",
        output_path="/home/zengkehan/voice/multisource_with_intensity",
        max_sources=3,
        fs=48000
        # (中文注释) 修改：移除 samples_len = 16000 参数
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

    # (中文注释) 移除滤波器组设计，因为在主循环中并未使用
    # filters, center_freqs, bandwidths = design_bandpass_filters(fs, 4)
    # ...

    # (中文注释) 新增：为(已注释的)全局统计初始化变量
    # sum_mean = 0.0
    # sum_sq_mean = 0.0
    # total_samples_processed = 0

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

            # (中文注释) 修改：移除 _prepare_audio_segment 中的 samples_len 参数
            def _prepare_audio_segment(audio, orig_fs, target_fs):
                audio = np.asarray(audio)
                if audio.size == 0:
                    # (中文注释) 修改：返回空数组，而不是固定长度的零
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
                        # (中文注释) 修改：如果重采样后长度为0，返回空数组
                        return np.zeros(0, dtype=np.float32)
                    audio = resample(audio, new_len)

                # (中文注释) 修改：移除补齐或截取到指定长度的逻辑
                # if len(audio) < samples_len:
                #     ...
                # audio = np.asarray(audio[:samples_len], dtype=np.float32)

                # (中文注释) 修改：仅确保类型正确
                audio = np.asarray(audio, dtype=np.float32)
                return audio

            # (中文注释) 修改：调用不含 samples_len 的预处理函数
            audio = _prepare_audio_segment(audio, orig_fs, fs)

            # (中文注释) 新增：如果音频处理后为空（例如原始文件为空或过短），则跳过此声源
            if audio.size == 0:
                print(f"  - 警告：文件 {os.path.basename(audio_path)} 处理后为空，跳过。")
                continue  # 跳过这个声源

            # TODO: 对声源信号进行缩放
            current_rms = np.sqrt(np.mean(audio ** 2))
            if current_rms > 0:
                target_rms = 0.3  # 目标RMS值，可以根据需要调整
                scaling = target_rms / current_rms
                audio *= scaling

            # 创建声源
            position, azimuth, elevation = generate_source_position(room_dimension, r_min=3, r_max=10)
            room.add_source(position)
            # (中文注释) 修改：pyroomacoustics 会自动处理不同长度的信号（以最长的为准）
            room.sources[-1].signal = audio  # 使用-1索引确保添加到正确的源
            source_signals.append(audio)
            source_positions.append(position)

            # 保存元数据 - 方位角和俯仰角
            metadata['sources'].append({
                'azimuth_deg': float(np.rad2deg(azimuth)),
                'elevation_deg': float(np.rad2deg(elevation)),
            })
            metadata['source_files'].append(os.path.basename(audio_path))

        # (中文注释) 新增：更新 num_sources，因为某些源可能已被跳过
        num_sources = len(source_signals)
        metadata['num_sources'] = num_sources  # 在元数据中明确记录最终的声源数

        # 计算每个声源的实际信号强度
        band_intensities = []
        for i in range(num_sources):
            intensities, center_freqs = calculate_source_intensity_by_freq(source_signals[i], source_positions[i],
                                                                           mic_positions, fs=fs)
            band_intensities.append(intensities)
        metadata['intensities'] = band_intensities

        # 模拟房间声学，获得多通道信号
        if num_sources == 0:
            # (中文注释) 修改：无声源时，不再生成 samples_len 长度，而是生成1秒的静音
            silence_len = fs  # 1 秒
            mixed_signal = np.zeros((mic_positions.shape[1], silence_len), dtype=np.float64)
        else:
            # 有声源时进行房间声学模拟
            # pyroomacoustics 会自动将混合信号的长度设置为最长源的长度
            room.simulate()
            mixed_signal = room.mic_array.signals

        # TODO: 噪音统一到后期添加
        # ... (噪声代码保持不变)

        # 保存多通道音频和标签元数据

        # 保存多通道音频和标签元数据
        room_audio_dir = os.path.join(output_base, "wavs", f"sample_{sample_idx}")
        os.makedirs(room_audio_dir, exist_ok=True)
        # (中文注释) 修改：保存为单个多通道 .wav 文件，而不是每个通道一个文件
        # 这样更易于管理和后续切片
        output_wav_path = os.path.join(room_audio_dir, "mix.wav")
        # soundfile 期望 (samples, channels) 格式，所以需要转置
        sf.write(output_wav_path, mixed_signal.T, fs)

        # (中文注释) 移除旧的单通道保存循环
        # for ch in range(mixed_signal.shape[0]):
        #     sf.write(f"{room_audio_dir}/channel_{ch}.wav", mixed_signal[ch], fs)

        with open(os.path.join(output_base, "metadata", f"sample_{sample_idx}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        # (中文注释) 修改：打印的日志现在包含信号的实际长度
        print(f"生成样本 {sample_idx + 1}/{num_samples}，包含 {num_sources} 个声源，长度: {mixed_signal.shape[1]} 采样点")

    #     # (中文注释) 修改：全局统计（如果启用）必须处理可变长度
    #     total_samples_in_mix = mixed_signal.size # (num_channels * num_samples)
    #     sum_mean += np.sum(mixed_signal) # 累加总和
    #     sum_sq_mean += np.sum(mixed_signal ** 2) # 累加平方和
    #     total_samples_processed += total_samples_in_mix
    #
    # (中文注释) 修改：全局统计（如果启用）必须使用
    # global_mean = sum_mean / total_samples_processed
    # global_std = np.sqrt(sum_sq_mean / total_samples_processed - global_mean ** 2)
    # np.save(os.path.join(output_base, f"global_mean.npy"), global_mean)
    # np.save(os.path.join(output_base, f"global_std.npy"), global_std)


if __name__ == '__main__':
    generate_multi_source_dataset(
        dataset_path="/home/kehan.zeng/DATA2/librispeech/LibriSpeech/test-clean",
        output_path="/home/kehan.zeng/DATA2/voice/mssl_libri_cone_noclip",  # (中文注释) 建议使用新路径
        num_samples=3000,
        room_dimension=(120, 120, 100),
        mic_length=0.12,  # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        max_sources=3
        # (中文注释) 修改：不再传递 samples_len
    )