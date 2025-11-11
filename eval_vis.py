#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-03-24 12:44
# @Author : 箴澄
# @Site : 评估model文件好不好，在测试集上运行代码，看看效果
# @File : eval.py
# @Software: PyCharm
from util import blue_red_heatmap, print_gradients, get_logger, blue_red_heatmap_old
from load_data import AudioDoADataset
from metric import heatmapLoss
from model.MultiSource_3DCNN_mapNet_revise import MultiSource3DCNNMapNetRevise
# from model.MultiSource_3DCNN_mapNet import MultiSource3DCNNMapNet

from torch.utils.data import DataLoader
import torch

# 忽略Matplotlib的警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import logging
logging.disable(logging.DEBUG)

def evaluate_model(dataset_path, model_path='sound_model.pth', splitflag="test"):
    # 1. 加载模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = MultiSource3DCNNMapNetRevise().to(device)
    # model = MultiSource3DCNNMapNet().to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    logger = get_logger("./coneplus_revise.log")

    # 2. 加载测试数据集
    test_dataset = AudioDoADataset(root_dir=dataset_path, split=splitflag)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

    # 3. 初始化存储
    # true_azimuth = []
    # true_elevation = []
    # pred_azimuth = []
    # pred_elevation = []
    # sumloss = []
    with (torch.no_grad()):
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = heatmapLoss(pred, y)
            logger.info(f"Test Loss: {loss.item():.6f}")
            blue_red_heatmap_old(y[0:16].cpu().detach().numpy() * 5, "gt")
            blue_red_heatmap_old(pred[0:16].cpu().detach().numpy() * 5, "pred")

            # sumloss.append(loss.item())

            # row_num = 4
            # col_num = 4
            # row_pred = []
            # row_gt = []
            # for h in range(row_num):
            #     col_pred = []
            #     col_gt = []
            #     for w in range(col_num):
            #      pred_slice = pred[h * col_num + w].cpu().detach().numpy()
            #      heatmap_slice = y[h * col_num + w].cpu().detach().numpy()
            #      pred_slice = np.pad(pred_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
            #      heatmap_slice = np.pad(heatmap_slice[2:-2, 2:-2], ((2, 2), (2, 2)), 'constant', constant_values=1)
            #      col_pred.append(pred_slice)
            #      col_gt.append(heatmap_slice)
            #     row_pred.append(np.concatenate(col_pred, axis=1))
            #     row_gt.append(np.concatenate(col_gt, axis=1))
            # result = np.concatenate(row_pred, axis=0)
            # gt = np.concatenate(row_gt, axis=0)
            # result_uint8 = (result * 255).astype('uint8')
            # cv2.imwrite("evalresult.png", result_uint8)
            # gt_uint8 = (gt * 255).astype('uint8')
            # cv2.imwrite("evalgt.png", gt_uint8)

def plot_gt(dataset_path):
    dataset = AudioDoADataset(root_dir=dataset_path, split="val")
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for _, y in test_loader:
        blue_red_heatmap_old(y[0:16].cpu().detach().numpy(), "gt")

if __name__ == "__main__":
    evaluate_model(r"/home/kehan.zeng/DATA2/voice/mssl_libri_cone/", "/home/kehan.zeng/DATA1/param/libri/revise_460.pth", splitflag="test")
    # evaluate_model(dataset_path="/home/kehan.zeng/DATA2/voice/multisource_with_freq_analysis",
    #                model_path="/home/kehan.zeng/DATA1/param/freq/freq_1500.pth", splitflag="test")
    # plot_gt("/home/kehan.zeng/DATA2/voice/multisource_normalized")

    # evaluate_model(dataset_path="/home/zengkehan/voice/multisource4eval_2",
    #                model_path="/home/zengkehan/ssl/mulsource_newloss_model791.pth")
    # evaluate_model(dataset_path="/home/zengkehan/voice/multisource4eval_3",
    #                model_path="/home/zengkehan/ssl/mulsource_newloss_model791.pth")