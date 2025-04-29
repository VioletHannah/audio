import torch
def heatmapLoss(pred, target):
    # pred: [B, 128, 128]，target: [B, 128, 128]
    mse = torch.nn.MSELoss()
    # 计算均方误差损失
    loss = mse(pred, target)
    return torch.mean(loss)

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
