#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/7 18:07
# @Author : 箴澄
# @Func：使用 TDOA 进行单声源定位
# @File : TDOA.py
# @Software: PyCharm

import os
import json
import math
import numpy as np
from scipy.optimize import minimize
import soundfile as sf

from traditional.tdoa.gcc_phat import gcc_phat
from NSDmultiSSLdata import generate_mic_array_positions

# 声学参数
SOUND_SPEED = 343.0  # m/s
FS = 16000  # 假设采样率为16kHz（根据实际数据调整）
INTERP_FACTOR = 16  # GCC-PHAT插值因子

# 生成8x8麦克风阵列坐标（中心为原点）
n_mics = 8
array_size = 0.2  # 阵列边长（米）
mic_positions = generate_mic_array_positions(n_mics, array_size, [50, 50, 50])

# 选择参考麦克风对（优化计算速度）
ref_mic = 28  # 选择中心附近的麦克风（索引28）
mic_pairs = [(j, ref_mic) for j in range(n_mics ** 2) if j != ref_mic]


def objective(params, pairs, tdoas, mics, c):
    """优化目标函数：最小化时延误差平方和"""
    theta, phi = params
    k = np.array([
        np.cos(theta) * np.cos(phi),
        np.sin(theta) * np.cos(phi),
        np.sin(phi)
    ])
    error = 0.0
    for (i, j), tau in zip(pairs, tdoas):
        dr = mics[:,j] - mics[:,i]
        # error += (np.dot(dr, k) / c - tau) ** 2
        error += (10*(np.dot(dr, k) - c * tau)) ** 2
    return error


def transfer_to_vector(azimuth, elevation):
    """将球坐标转换为单位向量，输入为弧度"""
    x = math.cos(azimuth) * math.cos(elevation)
    y = math.sin(azimuth) * math.cos(elevation)
    z = math.sin(elevation)
    return np.array([x, y, z])


def calculate_spatial_angle(pred_az, pred_el, true_az, true_el):
    """计算空间角度误差（弧度）"""
    v_pred = transfer_to_vector(pred_az, pred_el)
    v_true = transfer_to_vector(true_az, true_el)
    dot = np.clip(np.dot(v_pred, v_true), -1.0, 1.0)
    return math.acos(dot)


if __name__ == "__main__":
    # 主处理流程
    errors = []
    az_true_lst = []
    el_true_lst = []
    az_pred_lst = []
    el_pred_lst = []
    dataset_path = "/home/zengkehan/voice/speech_snr_30"
    for i in range(150):
        # 读取标签
        json_path = os.path.join(dataset_path, "metadata", f"sample_{i}.json")
        with open(json_path) as f:
            label = json.load(f)
        az_true = label['source_azimuth']
        el_true = label['source_elevation']
        az_true_lst.append(az_true)
        el_true_lst.append(el_true)

        # 读取所有麦克风信号
        sample_path = os.path.join(dataset_path, "wavs", f"sample_{i}")
        signals = []
        for mic in range(n_mics ** 2):
            sig, _ = sf.read(os.path.join(sample_path, f'channel_{mic}.wav'))
            signals.append(sig.astype(float))

        # 计算所有TDOA
        tdoas = []
        for k, j in mic_pairs:
            tau, _ = gcc_phat(signals[k], signals[j], fs=FS, interp=INTERP_FACTOR)
            tdoas.append(tau)
            # print(f"麦克风对 ({k}, {j}) 的时延：{tau:.6f}秒")

        # 优化方位角和俯仰角
        res = minimize(
            objective,
            x0=[math.pi, math.pi/4],
            args=(mic_pairs, tdoas, mic_positions, SOUND_SPEED),
            bounds=[(0, 2 * math.pi), (0, math.pi / 2)],
            method='L-BFGS-B'
        )
        az_pred, el_pred = res.x
        az_pred_lst.append(az_pred)
        el_pred_lst.append(el_pred)

        # 计算并存储误差
        errors.append(calculate_spatial_angle(az_pred, el_pred, az_true, el_true))

    with open("errors.txt", "w") as f:
        # 保存errors为JSON文件
        f.write(json.dumps(errors, indent=4))

        # 输出结果
        # logger.info(f"样本 {i + 1}/{1000}：")
        # logger.info(f"预测方位角：{np.degrees(az_pred):.2f}°，预测俯仰角：{np.degrees(el_pred):.2f}°")
        # logger.info(f"真实方位角：{np.degrees(az_true):.2f}°，真实俯仰角：{np.degrees(el_true):.2f}°")
        # logger.info(f"空间角度误差：{np.degrees(errors[-1]):.2f}°")
        # logger.info("----------------------------------------------------")
        # print(f"样本 {i + 1}/{500}：\n预测方位角：{np.degrees(az_pred):.2f}°，预测俯仰角：{np.degrees(el_pred):.2f}°")
        # print(f"真实方位角：{np.degrees(az_true):.2f}°，真实俯仰角：{np.degrees(el_true):.2f}°")
        # print("----------------------------------------------------")

    # 输出统计结果
    # acc1 = np.where(np.degrees(errors) < 5)[0].shape[0] / len(errors)
    # acc2 = np.where(np.degrees(errors) < 10)[0].shape[0] / len(errors)
    # acc3 = np.where(np.degrees(errors) < 15)[0].shape[0] / len(errors)
    # print(f"准确率（<5°）：{acc1:.2%}")
    # print(f"准确率（<10°）：{acc2:.2%}")
    # print(f"准确率（<15°）：{acc3:.2%}")
    # print(f"平均空间角度误差：{np.degrees(np.mean(errors)):.2f}度")
    # print(f"最大误差：{np.degrees(np.max(errors)):.2f}度")
    # print(f"误差标准差：{np.degrees(np.std(errors)):.2f}度")
    print(f"平均方位角误差：{np.degrees(np.mean(np.abs(np.array(az_true_lst) - np.array(az_pred_lst)))):.2f}°")
    print(f"平均俯仰角误差：{np.degrees(np.mean(np.abs(np.array(el_true_lst) - np.array(el_pred_lst)))):.2f}°")

    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    # 可视化误差分布
    # plot_joint_error_heatmap(az_true_lst, el_true_lst, az_pred_lst, el_pred_lst)

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10, 6), dpi=200)
    plt.hist(np.degrees(errors), bins=30, edgecolor='black', alpha=0.5)
    plt.xlabel("Spatial Angle Error (degrees)")
    plt.ylabel("Sample Count")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()


# def test_gcc_phat():
#     """测试GCC-PHAT函数"""
#     # 生成两个信号，第二个信号比第一个信号延迟5个采样点
#     test_sig1 = np.random.randn(16000)
#     test_sig2 = np.roll(test_sig1, 5)  # 制造5个样本的时延
#     tau, _ = gcc_phat(test_sig1, test_sig2, FS, INTERP_FACTOR)
#     print(f"测试GCC-PHAT时延：{tau:.6f}秒")
#     print(f"实际时延：{5 / FS:.6f}秒")

