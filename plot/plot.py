import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 配置参数
DATASET_PATH = '../voice/google_speech_commands'  # 数据集路径
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
MAX_FILES = 100  # 示例文件数量
BIN_WIDTH = 200  # 每个分箱200Hz


def generate_freq_bins():
    """生成200Hz带宽的频率分箱"""
    max_freq = SAMPLE_RATE // 2  # Nyquist频率
    bins = np.arange(0, max_freq + BIN_WIDTH, BIN_WIDTH)
    return bins


def analyze_frequency_distribution(freq_bins):
    """统计频率能量分布"""
    bin_energy = np.zeros(len(freq_bins) - 1, dtype=np.float64)

    # 获取文件列表
    file_list = []
    for root, _, files in os.walk(DATASET_PATH):
        file_list.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    if MAX_FILES:
        file_list = file_list[:MAX_FILES]

    # 预计算STFT频率点
    stft_freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

    # 处理所有文件
    for file_path in tqdm(file_list, desc='Processing'):
        try:
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mag = np.abs(D)

            # 将STFT频率映射到分箱
            freqs_indices = np.digitize(stft_freqs, freq_bins) - 1

            # 分箱累加能量
            for bin_idx in range(len(freq_bins) - 1):
                mask = (freqs_indices == bin_idx)
                bin_energy[bin_idx] += np.sum(mag[mask, :])

        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")

    # 计算百分比
    total_energy = np.sum(bin_energy)
    energy_percent = (bin_energy / total_energy) * 100
    return freq_bins, energy_percent


def plot_barchart(freq_bins, energy_percent):
    """绘制柱形图"""
    plt.figure(figsize=(15, 6))

    # 生成分箱标签
    labels = [f"{int(freq_bins[i])}-{int(freq_bins[i + 1])}Hz"
              for i in range(len(freq_bins) - 1)]

    # 创建柱形图
    bars = plt.bar(range(len(energy_percent)), energy_percent,
                   width=0.8,
                   edgecolor='steelblue',
                   linewidth=0.5,
                   color='skyblue')

    # 标注最高柱
    max_idx = np.argmax(energy_percent)
    bars[max_idx].set_color('indianred')
    plt.annotate(f'Peak: {energy_percent[max_idx]:.1f}%',
                 xy=(max_idx, energy_percent[max_idx]),
                 xytext=(max_idx, energy_percent[max_idx] + 2),
                 ha='center',
                 arrowprops=dict(arrowstyle='->', color='maroon'))

    # 坐标轴设置
    plt.xticks(range(len(energy_percent)), labels, rotation=45, ha='right')
    plt.ylabel('Energy Proportion (%)', fontsize=15)
    # plt.title(f'Frequency Energy Distribution ({BIN_WIDTH}Hz Bins)')
    plt.xticks(fontsize=15)
    plt.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        if height > 3:  # 只标注较大值
            plt.text(rect.get_x() + rect.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_hist_style(freq_bins, energy_percent):
    """直方图式柱形图绘制"""
    plt.figure(figsize=(15, 6))

    # 计算分箱参数
    bin_left = freq_bins[:-1]  # 分箱左边界
    bin_width = np.diff(freq_bins)  # 各分箱实际宽度
    bin_centers = bin_left + bin_width / 2  # 分箱中心点

    # 绘制柱形（左对齐、宽度精确匹配）
    bars = plt.bar(bin_left, energy_percent,
                   width=bin_width,
                   align='edge',  # 关键参数：柱形左对齐
                   edgecolor='navy',
                   linewidth=0.5,
                   color='skyblue')

    # 坐标轴标注
    plt.xticks(freq_bins, rotation=45)  # 刻度在分箱边界
    plt.xlim(freq_bins[0], freq_bins[-1])

    # 自动优化刻度密度
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))  # 主刻度每1kHz
    ax.xaxis.set_minor_locator(plt.MultipleLocator(200))  # 次刻度每200Hz

    # 标注设置
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Energy Proportion (%)', fontsize=15)
    # plt.title(f'Frequency Distribution')
    plt.xticks(fontsize=15)
    plt.grid(axis='y', alpha=0.3)

    # 添加中心点数值标签
    for center, percent in zip(bin_centers, energy_percent):
        if percent > 1:  # 仅标注显著分量
            plt.text(center, percent + 0.2, f'{percent:.1f}%',
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    freq_bins = generate_freq_bins()
    _, energy_percent = analyze_frequency_distribution(freq_bins)
    plot_hist_style(freq_bins, energy_percent)