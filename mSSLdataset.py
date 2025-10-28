#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/5 12:55
# @Author : 箴澄
# @File : mSSLdataset.py
# @Software: PyCharm
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
import soundfile as sf
from scipy.signal import resample, butter, filtfilt
import random
import json
import os
import glob

def generate_source_position(room_dimension, r_min=4, r_max=30):
    """
    随机生成一个声源位置
    :param room_dimension: 房间尺寸
    :return: 声源位置数组，方位角，俯仰角（弧度）
    """
    azimuth = np.deg2rad(np.random.uniform(0, 360))  # 方位角
    elevation = np.deg2rad(np.random.uniform(1, 91))
    r = np.random.uniform(r_min, r_max)  # 距离

    # 转换为笛卡尔坐标
    x = r * np.cos(elevation) * np.cos(azimuth) + room_dimension[0] / 2
    y = r * np.cos(elevation) * np.sin(azimuth) + room_dimension[1] / 2
    z = r * np.sin(elevation)

    final_elevation = elevation if np.rad2deg(elevation) >= 27 else np.deg2rad(27)
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

def calculate_source_intensity_over_time(source_signal, source_position, mic_positions, fs=48000, adc_range=1.0, sensitivity_mv_pa=50):
    """
    计算声源在各个时间窗口的信号强度
    :param source_signal: 源信号
    :param source_position: 声源位置 [x, y, z]
    :param mic_positions: 麦克风位置数组 (3, num_mics)
    :param fs: 采样率
    :return: 时间窗口强度列表，每个窗口的平均强度
    """
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
    window_length = 16000  # 样本数16000
    hop_size = int(0.1 * fs)  # 100ms的样本数
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

        b,a = butter(order, [lownorm, highnorm], btype='band')

        filters.append((b, a))
        bandwidths.append(bandwidth)

    return filters, center_freqs, bandwidths

from scipy.signal import butter, sosfiltfilt, stft
import warnings

def calculate_source_intensity_by_freq(source_signal, source_position, mic_positions,
                                       fs=48000, order=4,
                                       adc_range=1.0, sensitivity_mv_pa=50):
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

    # 定义中心频率（ISO 1/3倍频程标准）
    center_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    # 计算平均距离衰减
    distances = np.linalg.norm(source_position.reshape(3, 1) - mic_positions, axis=0)
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

            # 电压 -> 声压（Pa）
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


def calculate_source_intensity_by_freq_backup(source_signal, source_position, mic_positions, filters, fs=48000, adc_range=1.0, sensitivity_mv_pa=50):
    """
    计算声源在各个频段的信号强度
    :param source_signal: 源信号
    :param source_position: 声源位置 [x, y, z]
    :param mic_positions: 麦克风位置数组 (3, num_mics)
    :param filters: 滤波器组列表
    :param fs: 采样率
    :return: 各频段强度列表
    """
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
        mic_length=0.12, # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        dataset_path="path/to/nsynth-train/audio",
        output_path="/home/zengkehan/voice/multisource_with_intensity",
        max_sources=3
):
    # 配置输出路径
    output_base = output_path
    os.makedirs(os.path.join(output_base, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "metadata"), exist_ok=True)

    # 加载音频文件列表
    # audio_files = glob.glob(os.path.join(dataset_path, "*.flac"))
    # if not audio_files:
    #     raise ValueError("未找到AudioSet音频文件")
    # === 递归加载 LibriSpeech 的所有音频文件 ===
    audio_files = glob.glob(os.path.join(dataset_path, "**", "*.flac"), recursive=True)
    if not audio_files:
        raise ValueError(f"未在 {dataset_path} 下找到 LibriSpeech 音频文件")

    # 创建麦克风阵列
    mic_positions = generate_mic_array_positions(mic_num_per_line, mic_length, room_dimension)

    fs = 48000
    samples_len = 16000

    # 滤波器组设计
    # center_freqs = [125, 500, 2000, 8000, 16000]  # 中心频率 (Hz)
    # bandwidths = [200, 400, 2000, 6000, 8000]
    filters, center_freqs, bandwidths = design_bandpass_filters(fs, 4)
    filter_config = {
        'num_band': len(center_freqs),
        'center_freqs': center_freqs,
        'bandwidths': bandwidths,
        'order': 4
    }
    with open(os.path.join(output_base, "filter_config.json"), 'w') as f:
        json.dump(filter_config, f, indent=4)

    # 记录每个样本的功率信息
    sum_mean = 0
    sum_sq_mean = 0
    for sample_idx in range(num_samples):
        metadata = {'sources': [], 'source_files': [], 'intensities': []}
        # 创建房间，添加麦克风阵列
        room = pra.ShoeBox(room_dimension, fs=fs, absorption=1.0, max_order=0)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        # 随机生成声源数量（0-3），随后添加声源信息
        # num_sources = np.random.randint(0, max_sources + 1)
        num_sources = random.choices(list(range(max_sources + 1)), [0.1, 0.2, 0.3, 0.4])[0]  # 根据权重随机选择声源数量
        # 存储所有声源的原始信号和位置
        source_signals = []
        source_positions = []
        # 开始添加声源
        for i in range(num_sources):
            # 随机选择音频文件
            audio_path = random.choice(audio_files)
            audio, orig_fs = sf.read(audio_path)
            # 转为单声道
            if len(audio.shape) > 1:
                if audio.shape[1] == 2:
                    audio = np.mean(audio, axis=1)
                elif audio.shape[1] > 2:
                    audio = audio[:, 0]

            # 统一采样率并截取
            if fs != orig_fs:
                audio = resample(audio, int(48000 * len(audio) / fs))
            if len(audio) < samples_len:
                repeats = int(np.ceil(samples_len / len(audio)))
                audio = np.tile(audio, repeats)
            audio = audio[:samples_len] # 取前16000个采样点

            # 创建声源
            position, azimuth, elevation = generate_source_position(room_dimension, r_min=3, r_max=10)
            room.add_source(position)
            room.sources[i].signal = audio
            source_signals.append(audio)
            source_positions.append(position)

            # 保存元数据 - 方位角和俯仰角
            metadata['sources'].append({
                'azimuth_deg': float(np.rad2deg(azimuth)),
                'elevation_deg': float(np.rad2deg(elevation)),
            })
            metadata['source_files'].append(os.path.basename(audio_path))

        # 计算每个声源的实际信号强度
        band_intensities = []
        for i in range(num_sources):
            intensities = calculate_source_intensity_by_freq(
                source_signals[i],
                source_position=source_positions[i],
                mic_positions=mic_positions,
                filters=filters,
                fs=fs
            )
            band_intensities.append(intensities)
        metadata['intensities'] = band_intensities

        # 模拟房间声学，获得多通道信号
        if num_sources == 0:
            # 无生源时生成零信号
            mixed_signal = np.zeros((mic_positions.shape[1], samples_len), dtype=np.float32)
        else:
            # 有声源时进行房间声学模拟
            room.simulate()
            mixed_signal = room.mic_array.signals
        # 统一添加噪声，噪声功率为信号功率的1/20
        signal_power = np.mean(mixed_signal ** 2)
        if signal_power < 1e-10: # 无生源时
            noise_power = 0.01  # 默认噪声功率
        else:
            noise_power = signal_power / 20
        # 添加噪声
        noise = np.random.normal(0, np.sqrt(noise_power), mixed_signal.shape)
        mixed_signal += noise

        # 保存多通道音频和标签元数据
        room_dir = os.path.join(output_base, "wavs", f"sample_{sample_idx}")
        os.makedirs(room_dir, exist_ok=True)
        for ch in range(mixed_signal.shape[0]):
            sf.write(f"{room_dir}/channel_{ch}.wav", mixed_signal[ch], fs)
        with open(os.path.join(output_base, "metadata", f"sample_{sample_idx}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"生成样本 {sample_idx + 1}/{num_samples}，包含 {num_sources} 个声源")

        # 统计样本信息
        sum_mean += np.mean(mixed_signal)
        sum_sq_mean += np.mean(mixed_signal ** 2)

    global_mean = sum_mean / num_samples / (16*16*samples_len)
    global_std = np.sqrt(sum_sq_mean / num_samples / (16*16*samples_len) - global_mean ** 2)
    np.save(os.path.join(output_base, f"global_mean.npy"), global_mean)
    np.save(os.path.join(output_base, f"global_std.npy"), global_std)

if __name__ == '__main__':
    generate_multi_source_dataset(
        # dataset_path="/home/zengkehan/voice/audio/bal_train",
        dataset_path="/home/kehan.zeng/DATA2/voice/bal_train_segment",
        output_path="/home/kehan.zeng/DATA2/voice/multisource_with_freq_analysis",
        num_samples=10000,
        room_dimension=(120, 120, 100),
        mic_length=0.12,  # 8mm 麦克风间距 * 15 = 120mm
        mic_num_per_line=16,
        max_sources=3
    )