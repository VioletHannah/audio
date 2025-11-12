#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/11/11 16:00
# @Author : 箴澄
# @File : dataset_config.py
# @Software: PyCharm

FS = 48000  # 采样率

# 窗口（切片）参数
WINDOW_SIZE = 16000  # 约 333ms
OVERLAP = 0.5        # 50% 重叠
HOP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP)) # 步长

# Welch 频谱分析参数
WELCH_FREQ_BINS = 128  # 对应 util.py 的默认值, 用于热力图
WELCH_NPERSEG = 2048   # 对应 util.py 的默认值, 窗口