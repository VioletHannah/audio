import numpy as np
import soundfile as sf
import json
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy import signal
import os

# 设置路径
base_path = "/home/kehan.zeng/DATA2/voice/multisource_with_intensity"
sample_id = "sample_1"
wav_path = os.path.join(base_path, "wavs", sample_id)
metadata_path = os.path.join(base_path, "metadata", f"{sample_id}.json")

# 加载元数据
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# 加载音频数据
mic_signals = []
num_channels = 256  # 16x16阵列
for ch in range(num_channels):
    audio, fs = sf.read(os.path.join(wav_path, f"channel_{ch}.wav"))
    mic_signals.append(audio)
mic_signals = np.array(mic_signals)

print(f"加载样本 {sample_id}")
print(f"采样率: {fs} Hz")
print(f"信号长度: {mic_signals.shape[1] / fs:.2f} 秒")
print(f"通道数: {mic_signals.shape[0]}")
print(f"声源数量: {len(metadata['sources'])}")
for i, src in enumerate(metadata['sources']):
    print(f"声源 {i}: 方位角={src['azimuth_deg']:.2f}°, 俯仰角={src['elevation_deg']:.2f}°")


def music_doa_estimation(mic_signals, fs, mic_positions, wavelength, num_sources=None,
                         angle_range=np.arange(0, 360, 0.5)):
    """
    使用MUSIC算法进行方位角估计 (0-360度范围)

    参数:
    mic_signals: 麦克风阵列信号 (num_channels, num_samples)
    fs: 采样率
    mic_positions: 麦克风位置 (3, num_channels)
    wavelength: 信号波长
    num_sources: 声源数量，如果为None则自动估计
    angle_range: 扫描的角度范围 (0-360度)

    返回:
    doa_estimates: 估计的方位角
    music_spectrum: MUSIC空间谱
    """
    # 计算协方差矩阵
    num_channels, num_samples = mic_signals.shape
    Rxx = (mic_signals @ mic_signals.conj().T) / num_samples

    # 特征值分解
    eigenvalues, eigenvectors = eig(Rxx)
    # 对特征值和特征向量按特征值降序排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 如果未指定声源数量，则自动估计
    if num_sources is None:
        # 使用简单的阈值方法估计声源数量
        eig_ratio = eigenvalues / np.max(eigenvalues)
        num_sources = np.sum(eig_ratio > 0.1)  # 经验阈值

    # 划分噪声子空间
    noise_subspace = eigenvectors[:, num_sources:]

    # 计算MUSIC谱
    music_spectrum = np.zeros_like(angle_range, dtype=float)

    for i, theta in enumerate(angle_range):
        # 构建当前角度的导向矢量 (0-360度范围)
        theta_rad = np.deg2rad(theta)
        # 计算每个麦克风相对于参考点的相位延迟
        phase_delays = (mic_positions[0, :] * np.cos(theta_rad) +
                        mic_positions[1, :] * np.sin(theta_rad)) / wavelength
        a_theta = np.exp(-1j * 2 * np.pi * phase_delays)
        a_theta = a_theta.reshape(-1, 1)  # 转换为列向量

        # 计算MUSIC谱
        denominator = np.abs(a_theta.conj().T @ noise_subspace @ noise_subspace.conj().T @ a_theta)
        music_spectrum[i] = 1 / denominator.squeeze()

    # 寻找峰值
    peaks, _ = signal.find_peaks(music_spectrum, height=0.1 * np.max(music_spectrum), distance=5)
    # 选择最高的num_sources个峰值
    peak_values = music_spectrum[peaks]
    top_peaks = peaks[np.argsort(peak_values)[-num_sources:]]
    doa_estimates = angle_range[top_peaks]

    return doa_estimates, music_spectrum


# 麦克风阵列参数
mic_num_per_line = 16
mic_length = 0.12  # 阵列边长
spacing = mic_length / (mic_num_per_line - 1)  # 麦克风间距

# 生成麦克风位置 (与数据生成时一致)
mic_positions = np.zeros((3, mic_num_per_line * mic_num_per_line))
offset = (mic_num_per_line - 1) / 2
for i in range(mic_num_per_line):
    for j in range(mic_num_per_line):
        idx = i * mic_num_per_line + j
        mic_positions[0, idx] = (i - offset) * spacing
        mic_positions[1, idx] = (j - offset) * spacing
        mic_positions[2, idx] = 0

# 设置信号参数
c = 343  # 声速 (m/s)
f_center = 1000  # 中心频率 (Hz)
wavelength = c / f_center

# 应用MUSIC算法 (使用0-360度范围)
angle_range = np.arange(0, 360, 0.5)  # 扫描角度范围，步长0.5度
estimated_azimuths, music_spectrum = music_doa_estimation(
    mic_signals, fs, mic_positions, wavelength,
    num_sources=len(metadata['sources']),
    angle_range=angle_range
)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(angle_range, 10 * np.log10(music_spectrum / np.max(music_spectrum)), label='MUSIC Spectrum')
plt.xlabel('Azimuth Angle (Degree)')
plt.ylabel('Normalized Spectrum (dB)')
plt.title('DOA Estimation using MUSIC Algorithm (0-360°)')
plt.grid(True, linestyle='--', alpha=0.7)

# 标记真实方位角
true_azimuths = [src['azimuth_deg'] for src in metadata['sources']]
for i, az in enumerate(true_azimuths):
    plt.axvline(x=az, color='r', linestyle='--', alpha=0.7,
                label=f'True Azimuth {i + 1}: {az:.2f}°' if i == 0 else "")

# 标记估计方位角
for i, az in enumerate(estimated_azimuths):
    plt.axvline(x=az, color='g', linestyle='-.', alpha=0.7,
                label=f'Estimated Azimuth {i + 1}: {az:.2f}°' if i == 0 else "")

plt.legend()
plt.ylim(-30, 0)  # 设置纵轴范围
plt.xlim(0, 360)  # 设置横轴范围为0-360度
plt.show()

# 打印结果
print("\nMUSIC算法方位角估计结果:")
print(f"真实方位角: {true_azimuths}")
print(f"估计方位角: {estimated_azimuths}")

# 计算误差 (考虑角度循环性)
errors = []
for true_az in true_azimuths:
    min_error = min([abs(true_az - est_az) for est_az in estimated_azimuths])
    # 考虑角度循环性 (360度等于0度)
    min_error = min(min_error, 360 - min_error)
    errors.append(min_error)

print(f"方位角估计误差: {errors} 度")
print(f"平均误差: {np.mean(errors):.2f} 度")