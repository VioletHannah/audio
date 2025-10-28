#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/15 09:39
# @Author : 箴澄
# @Func：
# @File : plot_intensity.py
# @Software: PyCharm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def analyze_intensities(dataset_path, save_stats=False):
    """
    分析数据集中的声源强度分布
    :param dataset_path: 数据集路径
    """
    # 定位元数据目录
    metadata_dir = os.path.join(dataset_path, "metadata")
    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"元数据目录不存在: {metadata_dir}")

    # 收集所有JSON文件
    json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"未找到元数据文件: {metadata_dir}")
    print(f"找到 {len(json_files)} 个元数据文件")

    # 初始化统计变量
    all_intensities = []  # 所有强度值
    per_source_intensities = defaultdict(list)  # 按声源数量分组
    per_frequency_intensities = defaultdict(list)  # 按频率分组
    max_intensity = 0
    max_intensity_file = ""

    # 处理每个元数据文件
    for json_file in tqdm(json_files, desc="分析元数据"):
        file_path = os.path.join(metadata_dir, json_file)

        with open(file_path, 'r') as f:
            metadata = json.load(f)
            # 提取强度数据
            intensities = metadata.get('intensities', [])
            if not intensities:
                continue

            # 处理每个声源的强度
            num_sources = len(intensities)
            for intensity in intensities:
                # 更新全局最大值
                current_max = max(intensity)
                if current_max > max_intensity:
                    max_intensity = current_max
                    max_intensity_file = json_file

                # 收集所有强度值
                all_intensities.extend(intensity)

                # 按声源数量分组
                per_source_intensities[num_sources].extend(intensity)

                # 按频率分组
                for j in range(len(intensity)):
                    per_frequency_intensities[j].append(intensity[j])

    # 计算统计信息
    if not all_intensities:
        print("未找到强度数据")
        return

    all_intensities = np.array(all_intensities)

    print("\n===== 全局统计 =====")
    print(f"总强度值数量: {len(all_intensities):,}")
    print(f"最大值: {np.max(all_intensities):.6f} (出现在 {max_intensity_file})")
    print(f"最小值: {np.min(all_intensities):.6f}")
    print(f"平均值: {np.mean(all_intensities):.6f}")
    print(f"中位数: {np.median(all_intensities):.6f}")
    print(f"标准差: {np.std(all_intensities):.6f}")
    print(f"99%分位数: {np.percentile(all_intensities, 99):.6f}")
    print(f"99.9%分位数: {np.percentile(all_intensities, 99.9):.6f}")
    # 保存平均值和标准差到文件
    if save_stats:
        stats_file = os.path.join(dataset_path, "intensity_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Mean Intensity: {np.mean(all_intensities):.6f}\n")
            f.write(f"Std Intensity: {np.std(all_intensities):.6f}\n")
        print(f"\n统计结果已保存至: {stats_file}")

    # 按声源数量分组统计
    print("\n===== 按声源数量分组统计 =====")
    for num_sources, values in sorted(per_source_intensities.items()):
        arr = np.array(values)
        print(f"{num_sources} 个声源 - 样本数: {len(arr):,}, "
              f"最大值: {np.max(arr):.6f}, 平均值: {np.mean(arr):.6f}")

    # 按频率分组统计
    print("\n===== 按频率分组统计 =====")
    center_freq = [125, 500, 2000, 8000, 16000]  # 示例频率
    for freq_idx, values in sorted(per_frequency_intensities.items()):
        arr = np.array(values)
        print(f"频率 {center_freq[freq_idx]} - 样本数: {len(arr):,}, "
              f"最大值: {np.max(arr):.6f}, 平均值: {np.mean(arr):.6f}")

    # 绘制直方图
    plt.figure(figsize=(18, 6))

    # 全局分布
    plt.subplot(1, 3, 1)
    plt.hist(all_intensities, bins=200, log=False, alpha=0.7)
    plt.title('Global Intensity Distribution')
    plt.xlabel('Source Intensity (dB)')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="-")

    # 按声源数量分布
    plt.subplot(1, 3, 2)
    for num_sources, values in sorted(per_source_intensities.items()):
        plt.hist(values, bins=100, alpha=0.5, label=f'{num_sources} sources', log=False)

    plt.title('Intensities Distribution By Sources Numbers')
    plt.xlabel('Source Intensity (dB)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    # 按频率分组分布
    plt.subplot(1, 3, 3)
    for freq_idx, values in sorted(per_frequency_intensities.items()):
        if freq_idx < len(center_freq):
            plt.hist(values, bins=100, alpha=0.5, label=f'{center_freq[freq_idx]} Hz', log=False)

    plt.title('Intensities Distribution By Frequency')
    plt.xlabel('Source Intensity (dB)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()

    # 保存结果
    # output_file = os.path.join(dataset_path, "intensity_analysis.png")
    # plt.savefig(output_file, dpi=300)
    # print(f"\n分析结果已保存至: {output_file}")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    dataset_path = "/home/kehan.zeng/DATA2/voice/multisource_with_freq_analysis"
    analyze_intensities(dataset_path)