"""
test_heatmap_stable.py

功能：
- 使用数值稳定的 SOS 滤波（sosfiltfilt）计算多频带强度；
- 当遇到短时或低频可能失稳时，自动回退到基于 STFT 的能量估计；
- 将强度结果转换为 doap（alpha,beta,intensities）并调用 create_heatmap_multiband 绘制热力图；
- 比较 SOS 与 STFT 方法输出，打印并绘图以便调试。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, stft
from mSSLdataset import generate_source_position, generate_mic_array_positions
from load_data import get_alpha_beta_intensity, create_heatmap_multiband, heatmap_plot
import warnings

# ---------- 稳定的滤波器设计（返回 sos 列表 和 center_freqs） ----------
def design_sos_filters(fs, center_freqs=None, order=4):
    if center_freqs is None:
        center_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    sos_list = []
    for cf in center_freqs:
        lowcut = cf / np.sqrt(2)
        highcut = cf * np.sqrt(2)
        # 归一化
        nyq = fs / 2.0
        low = max(lowcut / nyq, 1e-6)   # 防止为0
        high = min(highcut / nyq, 0.9999)
        if low >= high:
            # 不合理频段（例如超过奈奎斯特），构造一个简单窄带替代
            sos = butter(order, [min(low, 0.4999), max(high, 0.5001)], btype='band', output='sos')
        else:
            sos = butter(order, [low, high], btype='band', output='sos')
        sos_list.append(sos)
    return sos_list, center_freqs

# ---------- 基于 sosfiltfilt 的带能量估计（数值稳定） ----------
def band_energy_sos(signal, sos, fs, sensitivity_mv_pa=50, adc_range=1.0, clip_thresh=1e6):
    # 双向滤波（零相位）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            filtered = sosfiltfilt(sos, signal)
        except Exception as e:
            # 若 sosfiltfilt 失败（极端数值），退回到 lfilter-like 用 sosfiltfilt on stabilized signal
            filtered = sosfiltfilt(sos, np.clip(signal, -1e3, 1e3))

    # 数值保护
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=clip_thresh, neginf=-clip_thresh)
    # 若仍然异常则裁剪
    if not np.all(np.isfinite(filtered)):
        filtered = np.clip(filtered, -clip_thresh, clip_thresh)

    # 转换为 Pa（保留原函数中的比例）
    p_actual = filtered * (adc_range * 1000.0) / sensitivity_mv_pa
    rms = np.sqrt(np.mean(p_actual ** 2)) if filtered.size > 0 else 0.0
    return rms

# ---------- 基于 STFT 的带能量估计（鲁棒后备方案） ----------
def band_energy_stft(signal, fs, center_freqs, band_width_factor=np.sqrt(2)):
    # 参数：短时傅里叶变换窗口长度采用 1024 或尽量大的2的幂
    nfft = 4096 if len(signal) >= 4096 else 1024
    f, t, Z = stft(signal, fs=fs, nperseg=min(1024, len(signal)), nfft=nfft, boundary=None)
    psd = np.mean(np.abs(Z) ** 2, axis=1)  # 平均得到每个频点的功率
    energies = []
    for cf in center_freqs:
        low = cf / band_width_factor
        high = cf * band_width_factor
        # 选择频点
        idx = np.where((f >= low) & (f <= high))[0]
        if idx.size == 0:
            energies.append(0.0)
        else:
            band_power = np.sum(psd[idx])
            # 转换为近似 RMS：sqrt(power)，但这里我们只需相对量，后续可log转换
            energies.append(np.sqrt(band_power))
    return np.array(energies)

# ---------- 统一接口：计算多频段强度（尝试 sos -> 若异常回退 stft） ----------
def calculate_band_intensities_safe(signal, src_pos, mic_positions, fs=48000, sensitivity_mv_pa=50):
    sos_filters, center_freqs = design_sos_filters(fs)
    # 先用 sos 计算
    rms_list = []
    problem_idx = []
    for i, sos in enumerate(sos_filters):
        rms = band_energy_sos(signal, sos, fs, sensitivity_mv_pa=sensitivity_mv_pa)
        # 检查异常（inf, nan, 非有限大，或极大），标记回退
        if not np.isfinite(rms) or rms > 1e6:
            rms_list.append(0.0)
            problem_idx.append(i)
        else:
            rms_list.append(rms)

    # 若有问题索引，使用 STFT 在这些频带上重新估计
    if problem_idx:
        stft_vals = band_energy_stft(signal, fs, center_freqs)
        for i in problem_idx:
            rms_list[i] = stft_vals[i]

    # 将 RMS -> SPL(dB) 的近似转换（如原函数），并应用距离衰减
    # 计算平均衰减（与 mSSLdataset 相同）
    distances = np.linalg.norm(src_pos.reshape(3,1) - mic_positions, axis=0)
    avg_atten = np.mean(1.0 / (distances + 1e-8))

    spls = []
    for rms in rms_list:
        # 防止零或极小值
        if rms <= 0:
            spl = 0.0
        else:
            spl = 20.0 * np.log10(max(rms / 20e-6, 1e-12))
        spls.append(spl * avg_atten)
    return np.array(spls), center_freqs

# ---------- 调试/对比绘图函数 ----------
def plot_compare(intens_sos, intens_stft, center_freqs):
    x = np.array(center_freqs)
    plt.figure(figsize=(8,4))
    plt.semilogx(x, intens_sos, label='sos-based (RMS->dB)')
    plt.semilogx(x, intens_stft, '--', label='stft-based')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative intensity (a.u. / dB-scale)')
    plt.title('Band intensities comparison')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend()
    plt.show()

# ---------- 主测试流程 ----------
if __name__ == "__main__":
    # 基本参数
    fs = 48000
    duration = 1.0   # 建议 >= 1s 以便低频稳定；你也可以尝试 0.333s 来测试稳健性
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)

    # 测试信号：1kHz 单频（振幅较小）
    freq = 1000.0
    amp = 0.1
    single_tone = amp * np.sin(2*np.pi*freq*t)

    # 也可以测试白噪声或混合信号
    white_noise = 0.02 * np.random.randn(len(t))
    mixed = 0.5 * single_tone + white_noise

    # 阵列与声源位置
    room_dim = (120,120,100)
    mic_num_per_line = 4
    mic_length = 0.12
    mic_positions = generate_mic_array_positions(mic_num_per_line, mic_length, room_dim)
    src_pos, azimuth, elevation = generate_source_position(room_dim, r_min=4, r_max=5)
    print(f"Source azimuth={np.degrees(azimuth):.2f}°, elevation={np.degrees(elevation):.2f}°")

    # 用 sos 与 stft 分别计算 band 强度
    spls_safe, center_freqs = calculate_band_intensities_safe(single_tone, src_pos, mic_positions, fs=fs)
    # 额外直接使用 STFT 估计全部频段用于对照
    spls_stft = band_energy_stft(single_tone, fs, center_freqs)

    print("中心频率：", center_freqs)
    print("sos->spl (稳定后):", np.round(spls_safe,2))
    print("stft amplitude (参考):", np.round(spls_stft,4))

    # 若你希望观察 sos 与 stft 的对比曲线
    plot_compare(spls_safe, spls_stft, center_freqs)

    # 组织为 doap（alpha,beta,intensities）——使用 load_data 中格式
    sources_meta = [{"azimuth_deg": np.degrees(azimuth), "elevation_deg": np.degrees(elevation)}]
    intensity_list = [spls_safe.tolist()]   # 注意类型要是 list of lists
    doap = get_alpha_beta_intensity(sources_meta, intensity_list)
    print("Converted doap:", doap)

    # 生成热力图并显示
    heatmap = create_heatmap_multiband(doap, grid_size=128, center_freqs=center_freqs)
    heatmap_plot(heatmap.numpy(), title=f"Stable Heatmap for {int(freq)} Hz tone")

    print("Done.")
