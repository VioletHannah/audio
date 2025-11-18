#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/25 00:42
# @Author : 箴澄
# @Func：一些函数
# @File : util.py
# @Software: PyCharm
import warnings

import numpy as np
from scipy.signal import resample, get_window
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import logging
import datetime

def calculate_source_intensity_welch_spectrum(
        source_signal,
        source_position,
        mic_positions,
        fs=48000,
        freq_bins=128,
        freq_range=(20, 16000),
        nperseg=2048,
        adc_range=1.0,
        sensitivity_mv_pa=500
):
    """
    使用Welch方法计算声源的全局频谱强度
    返回:
        intensities: ndarray shape (freq_bins,), 各频率 bin 的强度（dB SPL，经距离衰减加权）
        frequencies: ndarray shape (freq_bins,), 对应的频率值 (Hz)
    """
    from scipy.signal import welch
    import warnings

    # 计算距离衰减
    distances = np.linalg.norm(
        np.array(source_position).reshape(3, 1) - mic_positions,
        axis=0
    )
    avg_attenuation = np.mean(1.0 / (distances + 1e-8))

    # Welch方法计算PSD
    freqs_welch, psd_welch = welch(
        source_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window='hann',
        scaling='density'
    )

    # 选择频率范围
    fmin, fmax = freq_range
    freq_mask = (freqs_welch >= fmin) & (freqs_welch <= fmax)
    freqs_selected = freqs_welch[freq_mask]
    psd_selected = psd_welch[freq_mask]

    if len(freqs_selected) < 2:
        return np.zeros(freq_bins), np.logspace(np.log10(fmin), np.log10(fmax), freq_bins)

    # 对数频率重采样
    target_freqs = np.logspace(np.log10(fmin), np.log10(fmax), freq_bins)
    log_freqs = np.log10(freqs_selected)
    log_targets = np.log10(target_freqs)
    psd_resampled = np.interp(log_targets, log_freqs, psd_selected)

    # PSD -> RMS -> SPL
    bandwidth = fs / nperseg  # 频率分辨率
    rms_voltage = np.sqrt(psd_resampled * bandwidth)
    p_rms = rms_voltage * (adc_range * 1000.0) / sensitivity_mv_pa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spl = np.where(p_rms > 20e-6, 20 * np.log10(p_rms / 20e-6), 0.0)

    intensities = spl * avg_attenuation
    intensities = np.clip(intensities, 0, None)

    return intensities, target_freqs


def calculate_source_intensity_fft_global(
        source_signal,
        source_position,
        mic_positions,
        fs=48000,
        freq_bins=128,
        freq_range=(20, 16000),
        window='hann',
        adc_range=1.0,
        sensitivity_mv_pa=500
):
    """
    使用FFT计算声源的全局频谱强度

    参数:
        source_signal: ndarray, 输入信号 shape (n_samples,)
        source_position: (3,) 声源坐标
        mic_positions: (3, N) 麦克风坐标
        fs: 采样率
        freq_bins: 输出的频率bins数量
        freq_range: (fmin, fmax) 频率范围 Hz
        window: 窗函数类型 'hann', 'hamming', 'blackman'
        adc_range, sensitivity_mv_pa: 电声转换参数

    返回:
        intensities: ndarray shape (freq_bins,), 各频率bin的强度
        frequencies: ndarray shape (freq_bins,), 对应的频率值(Hz)
    """

    # 1. 计算距离衰减
    distances = np.linalg.norm(
        np.array(source_position).reshape(3, 1) - mic_positions,
        axis=0
    )
    avg_attenuation = np.mean(1.0 / (distances + 1e-8))

    # 2. 应用窗函数减少频谱泄漏
    n_samples = len(source_signal)
    window_func = get_window(window, n_samples)
    windowed_signal = source_signal * window_func

    # 3. 计算FFT（实信号使用rfft更高效）
    fft_result = np.fft.rfft(windowed_signal)
    fft_freqs = np.fft.rfftfreq(n_samples, 1 / fs)

    # 4. 计算功率谱密度（PSD）
    # PSD = |FFT|^2 / N，归一化
    psd = (np.abs(fft_result) ** 2) / n_samples

    # 补偿窗函数的能量损失
    window_power = np.sum(window_func ** 2) / n_samples
    psd = psd / window_power

    # 5. 选择感兴趣的频率范围
    fmin, fmax = freq_range
    freq_mask = (fft_freqs >= fmin) & (fft_freqs <= fmax)
    freqs_selected = fft_freqs[freq_mask]
    psd_selected = psd[freq_mask]

    if len(freqs_selected) < 2:
        # 如果没有有效频率，返回零
        return np.zeros(freq_bins), np.logspace(np.log10(fmin), np.log10(fmax), freq_bins)

    # 6. 重采样到对数频率网格（更符合感知）
    # 对数频率采样：低频密集，高频稀疏
    target_freqs = np.logspace(
        np.log10(fmin),
        np.log10(fmax),
        freq_bins
    )

    # 使用对数插值（在对数域内线性插值）
    log_freqs_selected = np.log10(freqs_selected)
    log_target_freqs = np.log10(target_freqs)

    # 插值PSD到目标频率
    psd_resampled = np.interp(
        log_target_freqs,
        log_freqs_selected,
        psd_selected
    )

    # 7. 转换为RMS电压
    # PSD是功率，RMS = sqrt(PSD)
    rms_voltage = np.sqrt(psd_resampled)

    # 8. 电压 -> 声压 (Pa)
    p_rms = rms_voltage * (adc_range * 1000.0) / sensitivity_mv_pa

    # 9. 转换为声压级 (dB SPL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spl = np.where(
            p_rms > 20e-6,
            20 * np.log10(p_rms / 20e-6),
            0.0
        )

    # 10. 应用距离衰减
    intensities = spl * avg_attenuation
    intensities = np.clip(intensities, 0, None)

    return intensities, target_freqs


def compare_spectrum_methods(signal, position, mics, fs=48000):
    """对比不同频谱方法的结果"""
    import matplotlib.pyplot as plt

    # FFT方法
    int_fft, freq_fft = calculate_source_intensity_fft_global(
        signal, position, mics, fs=fs, freq_bins=128
    )
    print(freq_fft.shape)

    # Welch方法
    int_welch, freq_welch = calculate_source_intensity_welch_spectrum(
        signal, position, mics, fs=fs, freq_bins=128
    )

    plt.figure(figsize=(12, 5))
    plt.semilogx(freq_fft, int_fft, 'b-', label='FFT', linewidth=1.5)
    plt.semilogx(freq_welch, int_welch, 'r--', label='Welch', linewidth=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity (dB SPL)')
    plt.title('Spectrum Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

def visualize_continuous_spectrum(intensities, frequencies, title="Continuous Spectrum"):
    """可视化连续频谱"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 线性尺度
    ax1.plot(frequencies, intensities, 'b-', linewidth=1.5)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Intensity (dB SPL)', fontsize=12)
    ax1.set_title(f'{title} - Linear Scale', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 对数尺度
    ax2.semilogx(frequencies, intensities, 'r-', linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz, log scale)', fontsize=12)
    ax2.set_ylabel('Intensity (dB SPL)', fontsize=12)
    ax2.set_title(f'{title} - Log Scale', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

def prepare_audio_segment(audio, orig_fs, target_fs, samples_len):
    """
    预处理音频片段，统一单声道，长度和采样率
    1. 转为单声道
    2. 统一采样率
    3. 补齐或截取到指定长度
    4. 返回处理后的音频片段
    """
    audio = np.asarray(audio)
    if audio.size == 0:
        return np.zeros(samples_len, dtype=np.float32)
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
            new_len = samples_len
        audio = resample(audio, new_len)
    # 补齐或截取到指定长度
    if len(audio) < samples_len:
        repeats = int(np.ceil(samples_len / len(audio)))
        audio = np.tile(audio, repeats)
    audio = np.asarray(audio[:samples_len], dtype=np.float32)
    return audio

def rms_scaling(audio, target_rms=1.0):
    # 将声源信号放大到1.0
    # param - target_rms: 目标RMS值，可以根据需要调整
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        scaling = target_rms / current_rms
        audio *= scaling
    return audio


def heatmap_plot(heatmap, title="Heatmap", absflag=False):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    if absflag:
        plt.imshow(heatmap, cmap='jet', origin='lower', vmin=0, vmax=1)
    else:
        plt.imshow(heatmap, cmap='jet', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.xlabel('Beta (degrees)')
    plt.ylabel('Alpha (degrees)')
    plt.show()


def get_logger(filename=None):
    # 如果未指定文件名，使用当前时间作为默认名
    if not filename:
        cnt_time = datetime.datetime.now()
        filename = f'./{cnt_time.strftime("%Y%m%d_%H%M%S")}.log'

    logger = logging.getLogger()
    # 避免重复添加处理器
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 文件处理器（使用传入的文件名）
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def azimuth_elevation_to_alpha_beta(azimuth, elevation):
    """
    将方位角和俯仰角转换为alpha和beta
    :param azimuth: 方位角（以度为单位）与 x 轴正方向的夹角（在 xy 平面上）
    :param elevation: 俯仰角（以度为单位）与 xy 平面的夹角（z 方向的仰角）
    :return: alpha和beta（z-x夹角和z-y夹角，以度为单位）
    """

    azi_rad = np.radians(azimuth)
    ele_rad = np.radians(elevation)
    tan_phi = np.tan(ele_rad)

    # 计算k
    k = np.where(tan_phi == 0, np.inf, 1 / tan_phi)  # 处理tan_phi=0的情况

    # 计算tan_alpha和tan_beta
    tan_alpha = np.cos(azi_rad) / np.tan(ele_rad)
    tan_beta = np.sin(azi_rad) / np.tan(ele_rad)

    # 计算alpha和beta
    alpha_rad = np.arctan(tan_alpha)
    beta_rad = np.arctan(tan_beta)

    # 转换为角度
    alpha = np.degrees(alpha_rad)
    beta = np.degrees(beta_rad)

    return alpha, beta

def alpha_beta_to_azimuth_elevation(alpha, beta):
    """
    将alpha和beta转换为方位角和俯仰角
    :param alpha: z-x夹角角度
    :param beta: z-y夹角角度
    :return: 方位角和俯仰角（以度为单位）
    """

    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    tan_alpha = np.tan(alpha_rad)
    tan_beta = np.tan(beta_rad)

    # 计算theta（方位角）
    theta_rad = np.arctan2(tan_beta, tan_alpha)  # 自动处理象限

    # 计算phi（俯仰角）
    k = np.sqrt(tan_alpha ** 2 + tan_beta ** 2)
    phi_rad = np.where(k == 0, np.pi / 2, np.arctan(1 / k))  # 处理k=0的情况

    # 转换为角度
    azimuth = np.degrees(theta_rad) % 360
    elevation = np.degrees(phi_rad)

    return azimuth, elevation

def search_source_position(heatmap, thresh=0.5):
    """
    在热图中搜索声源位置
    参数：
    - heatmap: 形状为[128, 128]的热图
    - thresh: 阈值
    返回：
    - 源位置张量，形状为[num, 2]
    """
    source_positions = []
    # 找到大于阈值的点
    while np.max(heatmap) > thresh:
        index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # 计算声源位置
        alpha = np.clip(index[0] - 63, -63, 64)
        beta = np.clip(index[1] - 63, -63, 64)
        azimuth, elevation = alpha_beta_to_azimuth_elevation(alpha, beta)
        source_positions.append([azimuth, elevation])
        # 将最大值及其周围置为0，避免重复搜索
        c = 3
        heatmap[index[0]-c:index[0]+c+1, index[1]-c:index[1]+c+1] = 0
    return source_positions

def blue_red_heatmap(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    if not isinstance(data, np.ndarray) or data.shape != (16, 128, 128):
        raise ValueError("输入 data 必须是形状 (16,128,128) 的 numpy 数组")

    # 修改1：定义新的颜色映射（蓝 → 白 → 红）
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝(-1) → 白(0) → 红(1)
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)

    # 修改2：调整标准化范围到 [-1, 1]
    norm = Normalize(vmin=-1, vmax=1)

    # 绘图
    fig, axs = plt.subplots(4, 4, figsize=(10, 8),
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.suptitle(title, fontsize=15, y=0.92)

    for idx, ax in enumerate(axs.flat):
        # 修改3：使用新定义的 cmap 和 norm
        im = ax.imshow(data[idx], cmap=cmap, norm=norm,
                       origin='lower', aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes((0.90, 0.12, 0.015, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs

def blue_red_heatmap_old(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    if not isinstance(data, np.ndarray) or data.shape != (16, 128, 128):
        raise ValueError("输入 data 必须是形状 (16,128,128) 的 numpy 数组")

    # 配色和标准化
    colors = [(1,1,1), (0.5,0.5,1), (1,0,0)]  # 白 → 蓝 → 红
    cmap = LinearSegmentedColormap.from_list("white_blue_red", colors, N=256)
    norm = Normalize(vmin=0, vmax=1)

    # 绘图
    fig, axs = plt.subplots(4,4, figsize=(10, 8),
                            gridspec_kw={'wspace':0.05,'hspace':0.05})
    fig.suptitle(title, fontsize=15, y=0.92)

    for idx, ax in enumerate(axs.flat):
        im = ax.imshow(data[idx], cmap='jet', norm=norm,
                       origin='lower', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes((0.90, 0.12, 0.015, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs

def blue_red_heatmap_new(data=None, title="4×4 Gaussian-3-point Heatmaps", save_path=None):
    """
    生成16张 128×128 的三点高斯热图，4×4 网格显示。
    支持负值到正值的范围，颜色映射：负值(白→蓝)，0值(蓝)，正值(蓝→红)
    """
    if not isinstance(data, np.ndarray) or data.shape != (16, 128, 128):
        raise ValueError("输入 data 必须是形状 (16,128,128) 的 numpy 数组")

    # 计算全局归一化范围（确保包含0）
    vmin = min(data.min(), 0)  # 确保包含负值
    vmax = max(data.max(), 0)  # 确保包含正值
    mid = (0 - vmin) / (vmax - vmin)  # 0值在归一化后的位置

    # 自定义颜色映射：负值(白→蓝)，0值(蓝)，正值(蓝→红)
    cdict = {
        'red':   [(0.0,   1.0, 1.0),   # 起点：白色 (R=1)
                 (mid,   0.5, 0.5),   # 零点：蓝色 (R=0.5)
                 (1.0,   1.0, 1.0)],  # 终点：红色 (R=1)
        'green': [(0.0,   1.0, 1.0),   # 起点：白色 (G=1)
                 (mid,   0.5, 0.5),   # 零点：蓝色 (G=0.5)
                 (1.0,   0.0, 0.0)],  # 终点：红色 (G=0)
        'blue':  [(0.0,   1.0, 1.0),   # 起点：白色 (B=1)
                 (mid,   1.0, 1.0),   # 零点：蓝色 (B=1)
                 (1.0,   0.0, 0.0)]   # 终点：红色 (B=0)
    }
    cmap_custom = LinearSegmentedColormap('custom_blue_red', cdict)
    norm = Normalize(vmin, vmax)

    # 绘图
    fig, axs = plt.subplots(4,4, figsize=(10, 8),
                           gridspec_kw={'wspace':0.05,'hspace':0.05})
    fig.suptitle(title, fontsize=15, y=0.92)

    for idx, ax in enumerate(axs.flat):
        im = ax.imshow(data[idx], cmap=cmap_custom, norm=norm,
                      origin='lower', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])

    # 统一色条
    cax = fig.add_axes((0.90, 0.12, 0.015, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.yaxis.label.set_size(15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item():.4f}")
        else:
            print(f"{name} grad: None")

def apply_gaussian_filter_with_preserved_peak(data, sigma=1.0, kernel_size=5):
    """
    应用高斯模糊同时保持峰值高度
    :param data: 输入数据 (128x128 数组)
    :param sigma: 高斯核标准差 (控制平滑程度)
    :param kernel_size: 高斯核大小 (控制影响范围)
    :return: 处理后的数据
    """
    # 创建高斯核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-distance ** 2 / (2 * sigma ** 2))

    # 归一化核（保持峰值高度）
    kernel /= kernel[center, center]

    # 应用卷积
    from scipy.ndimage import convolve
    return convolve(data, kernel, mode='constant', cval=0.0)