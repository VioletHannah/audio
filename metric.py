import numpy as np
import torch
import math

def heatmapBiLoss(pred, target, threshold=0.5, alpha=20):
    # 二分类loss pred: [B, 128, 128]，target: [B, 128, 128]
    # 计算二分类交叉熵损失
    loss = 0
    for i in range(pred.shape[0]):
        pred_i = pred[i].unsqueeze(0)
        target_i = target[i].unsqueeze(0)
        # 计算二分类交叉熵损失
        bce_loss = torch.nn.BCELoss()(pred_i, target_i)
        # 计算加权损失
        weight = torch.where(target_i > threshold, alpha, 1.0)
        err = (pred_i - target_i) ** 2
        weighted_loss = (err * weight).sum() / pred_i.shape[0]
        # 将加权损失与二分类交叉熵损失相加
        loss += bce_loss + weighted_loss
    # 返回平均损失
    return loss / pred.shape[0]

def focalLoss(pred, target, alpha=0.85, gamma=2):
    # 计算P_t
    p_t = torch.where(target > 0.5, pred, 1 - pred)
    # 计算基础交叉熵损失
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    # 计算调制因子
    modulating_factor = (1 - p_t) ** gamma
    # 计算类别权重
    class_weight = torch.where(target > 0.5, alpha, 1 - alpha)
    # 计算加权损失
    weighted_loss = class_weight * modulating_factor * bce_loss
    # 返回平均损失
    return weighted_loss.mean()

def heatmapLoss(pred, target, threshold=0.5, alpha=20):
    # pred: [B, 128, 128]，target: [B, 128, 128]
    # weight = torch.where(target > threshold, alpha, 1.0)
    err = (pred - target) ** 2
    return err.sum() / pred.shape[0]
    # return (err * weight).sum() / pred.shape[0]

    # mse = torch.nn.MSELoss()
    # # 计算均方误差损失
    # loss = mse(pred, target)
    # return torch.mean(loss)

def transfer_to_vector(azimuth, elevation):
    """
    将方位角和俯仰角转换为三维向量（使用PyTorch操作保持梯度）
    """
    x = torch.cos(azimuth) * torch.cos(elevation)
    y = torch.sin(azimuth) * torch.cos(elevation)
    z = torch.sin(elevation)
    return torch.stack([x, y, z], dim=-1)  # 保持张量维度


def calculate_spatial_angle(label_azimuth, label_elevation, true_azimuth, true_elevation):
    # 同时处理批量数据
    label_vector = transfer_to_vector(label_azimuth, label_elevation)
    true_vector = transfer_to_vector(true_azimuth, true_elevation)

    # 批量点积
    dot_product = (label_vector * true_vector).sum(dim=-1)

    # 确保数值稳定性（使用PyTorch操作）
    dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6)

    # 计算空间角（保持梯度）
    return torch.acos(dot_product)


def newAngleLoss(pred, target, evaluate=False):
    """
    修改后的损失函数
    pred: [B, 2]（弧度），需要梯度
    target: [B, 2]（弧度）
    """
    # 直接处理整个批量，避免循环
    spatial_angles = calculate_spatial_angle(
        pred[:, 0],  # 预测方位角
        pred[:, 1],  # 预测俯仰角
        target[:, 0],  # 真实方位角
        target[:, 1]  # 真实俯仰角
    )
    if evaluate:
        return spatial_angles

    return spatial_angles.mean()

#!
def AngleLoss(pred, target):
    # pred: [B, 2]（弧度），target: [B, 2]（弧度）

    theta_pred, phi_pred = pred[:, 0], pred[:, 1]
    theta_target, phi_target = target[:, 0], target[:, 1]

    # 处理theta的360°周期性（正确周期为2π）
    theta_diff = torch.abs(theta_pred - theta_target) % (2 * torch.pi)
    theta_loss = torch.min(theta_diff, 2 * torch.pi - theta_diff)  # 修正周期为2π

    phi_loss = torch.abs(phi_pred - phi_target)

    # 将弧度转换为角度
    deg_factor = 180.0 / torch.pi
    theta_loss_deg = theta_loss * deg_factor
    phi_loss_deg = phi_loss * deg_factor

    # 返回角度损失的平均值
    return (theta_loss_deg + phi_loss_deg).mean()
