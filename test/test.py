"""
测试频率强度计算及热力图生成脚本
"""

import numpy as np
import torch
from scipy.signal import filtfilt
import soundfile as sf
import librosa

from mSSLdataset import (
    generate_source_position,
    generate_mic_array_positions,
    design_bandpass_filters,
    calculate_source_intensity_by_freq,  # 使用改进版频率强度计算函数
)
from load_data import get_alpha_beta_intensity, create_heatmap_multiband, heatmap_plot
from util import azimuth_elevation_to_alpha_beta

# 加载音频文件
# audio_path = "/home/kehan.zeng/DATA2/voice/bal_train/7NF2kcEfMBI.flac"
# audio_data, fs = librosa.load(audio_path, sr=48000)
# print(f"音频采样率: {fs}, 音频时长: {len(audio_data)/fs:.2f}秒")
# print(f"音频数据类型: {audio_data.dtype}, 音频数据形状: {audio_data.shape}")
# signal = audio_data[:48000]  # 截取1秒音频

# 可选：使用正弦信号进行测试
fs = 48000
duration = 1.0
freq = 1000
amplitude = 0.05
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = amplitude * np.sin(2 * np.pi * freq * t)
# signal = signal[:16000]  # 截取约0.333秒


# 配置房间和麦克风阵列参数
room_dim = (120, 120, 100)
mic_num_per_line = 4
mic_length = 0.12  # 麦克风阵列长度
mic_positions = generate_mic_array_positions(mic_num_per_line, mic_length, room_dim)

# 随机生成声源位置
source_pos, azimuth, elevation = generate_source_position(room_dim, r_min=4, r_max=5)
print(f"声源位置: {source_pos}")
print(f"方位角={np.degrees(azimuth):.2f}°, 俯仰角={np.degrees(elevation):.2f}°")

# 设计带通滤波器组
filters, center_freqs, _ = design_bandpass_filters(fs, order=4)

# 打印各频段滤波后的信号RMS值
for idx, (b, a) in enumerate(filters):
    filtered_signal = filtfilt(b, a, signal)
    print(f"{center_freqs[idx]:>6} Hz RMS: {np.sqrt(np.mean(filtered_signal **2)):.6f}")


# 调试：计算距离衰减信息
print("\n平均距离衰减校验：")
distances = np.linalg.norm(source_pos.reshape(3, 1) - mic_positions, axis=0)
print(f"  平均距离: {np.mean(distances):.2f}m, 最小距离: {np.min(distances):.2f}m")
print(f"  平均衰减因子: {np.mean(1/(distances + 1e-8)):.6f}")

# 调用改进版的频率强度计算函数
# 注意：改进版返回 (强度列表, 中心频率列表) 元组
band_intensities, calc_center_freqs = calculate_source_intensity_by_freq(
    signal,
    source_position=source_pos,
    mic_positions=mic_positions,
    fs=fs,  # 改进版不需要传入filters参数，内部已处理
    order=4
)


# 修正异常值（防止nan或inf）
# band_intensities = np.nan_to_num(band_intensities, nan=0.0, posinf=0.0, neginf=0.0)
# band_intensities = np.clip(band_intensities, 0, 120)  # 限制dB范围到合理区间

# 打印各频段强度结果
print("\n各频段强度 (裁剪后):")
for freq, intensity in zip(calc_center_freqs, band_intensities):
    print(f"{freq:>6} Hz: {intensity:.2f}")

# 生成方向能量图(DOAP)和热力图
sources = [{"azimuth_deg": np.degrees(azimuth), "elevation_deg": np.degrees(elevation)}]
intensity_list = [band_intensities]
doap = get_alpha_beta_intensity(sources, intensity_list)

heatmap = create_heatmap_multiband(doap, grid_size=128, center_freqs=calc_center_freqs)
# 可选：显示热力图
# heatmap_plot(heatmap.numpy(), title="单声源多频段方向热力图", absflag=True)

print("\n热力图生成完成（已防溢出）")