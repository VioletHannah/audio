#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/8/14 23:18
# @Author : 箴澄
# @Func：
# @Site : 
# @File : plot_loss.py
# @Software: PyCharm
import re
import matplotlib.pyplot as plt


def parse_log_file(file_path):
    epochs = []
    losses = []

    # 正则表达式匹配示例： "Epoch 25, Average Loss: 102.96255458484997"
    pattern = r'Epoch (\d+), Average Loss: (\d+\.\d+|\d+)'

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)

    return epochs, losses


def plot_loss_curve(epochs, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)

    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 自动调整Y轴范围，在最大值上方留10%空白
    y_max = max(losses)
    plt.ylim(0, y_max * 1.1)

    plt.tight_layout()
    # plt.savefig('loss_curve_revise.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    log_file = "spl_train.log"

    try:
        epochs, losses = parse_log_file(log_file)

        if not epochs:
            print("未找到包含Average Loss的日志记录")
        else:
            print(f"成功解析 {len(epochs)} 条记录")
            print("Epoch示例:", epochs[:5])
            print("Loss示例:", losses[:5])

            plot_loss_curve(epochs, losses)
            print("损失曲线已保存为 loss_curve.png")

    except FileNotFoundError:
        print(f"错误：文件 {log_file} 未找到")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")