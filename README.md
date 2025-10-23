# 基于3D CNN的多声源定位系统

使用3D卷积神经网络和麦克风阵列实现多声源空间定位的深度学习方案。

## 项目简介

本项目致力于实现一个端到端的多声源定位系统，能够同时检测和定位3D空间中的多个声源（0-3个）。系统通过16×16平面麦克风阵列采集音频数据，利用3D CNN处理时空特征，最终输出128×128的空间热力图。

### 主要特性

- 多声源定位：同时检测0-3个声源
- 3D CNN架构：高效提取时空音频特征
- 多频段分析：10个频段（31.5Hz - 16kHz）的强度分析
- 房间声学仿真：基于pyroomacoustics的真实模拟
- 热力图输出：直观的空间可视化结果

## 环境依赖

```bash
python >= 3.8
torch >= 1.12.0
numpy, scipy, soundfile
pyroomacoustics >= 0.6.0
matplotlib
```

建议使用GPU进行训练（8GB+ 显存）

## 工作流程

### 1. 数据集生成

使用NSynth或AudioSet音频数据生成多声源定位数据集：

```bash
python mSSLdataset.py
```

**主要参数：**
- `dataset_path`: 源音频文件路径
- `output_path`: 数据集输出路径
- `num_samples`: 生成样本数量（默认10000）
- `max_sources`: 最大声源数（默认3）

**生成的数据结构：**
```
output_path/
├── wavs/              # 多通道音频文件
│   ├── sample_0/
│   │   ├── channel_0.wav
│   │   └── ...
├── metadata/          # 标签文件（方位角、俯仰角、强度）
│   ├── sample_0.json
│   └── ...
└── filter_config.json
```

### 2. 模型训练

```bash
python train.py --data_dir /path/to/dataset \
                --batch_size 32 \
                --epochs 100 \
                --lr 0.001
```

### 3. 模型评估

```bash
python eval.py --model_path checkpoints/best_model.pth \
               --data_dir /path/to/dataset
```

### 4. 结果可视化

```bash
python eval_vis.py --model_path checkpoints/best_model.pth \
                   --sample_dir /path/to/sample \
                   --output_dir visualizations/
```

## 项目结构

```
.
├── mSSLdataset.py              # 数据集生成
├── load_data.py                # 数据加载器
├── MultiSource_3DCNN_mapNet.py # 模型定义
├── util.py                     # 工具函数
├── train.py                    # 训练脚本
├── eval.py                     # 评估脚本
└── eval_vis.py                 # 可视化脚本
```

## 模型架构

```
输入: [Batch, 1, 16000, 8, 8]  (时间序列 × 麦克风网格)
  ↓
3D卷积层 (3个stack，逐步提取特征)
  ↓
特征提取: [Batch, 1024, 8, 8]
  ↓
2D反卷积上采样 (4层，逐步恢复分辨率)
  ↓
输出: [Batch, 128, 128]  (空间热力图)
```

## 技术说明

### 坐标系统

- **球坐标**：方位角θ (0-360°)、俯仰角φ (0-90°)
- **Alpha-Beta表示**：α、β ∈ [-63°, 64°]，映射到128×128网格

### 数据增强

- 随机通道增益变化（0.8-1.2倍）
- 随机通道丢弃（20%概率，模拟麦克风故障）

### 频段分析

10个1/3倍频程频段，不同频段使用自适应高斯平滑：
- 高频：更尖锐的定位
- 低频：更宽泛的定位
