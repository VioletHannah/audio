import torch
import numpy as np
from load_data import AudioDoADataset, create_heatmap_multiband
from util import search_source_position, heatmap_plot
from eval import loc_source_position
from traditional.srp.SRP4mulssl import match_sources


dataset_root = "/home/kehan.zeng/DATA2/voice/mssl_libri"

dataset = AudioDoADataset(
    root_dir=dataset_root,
    split="train",
    heatmap_label=True,
    augm=False
)

# 读取一个样本
audio, heatmap = dataset[0]     # heatmap shape = [128,128]
heatmap_np = heatmap.numpy()

print("Loaded audio:", audio.shape)
print("Loaded heatmap:", heatmap_np.shape)

pred_pos = loc_source_position(heatmap_np.copy(), 0.2)
# 将预测的(alpha, beta)转换为(azim, elev)
from util import alpha_beta_to_azimuth_elevation
pred_pos = [alpha_beta_to_azimuth_elevation(alpha, beta) for alpha, beta in pred_pos]

print("Predicted positions (alpha,beta converted to azim,elev):")
print(pred_pos)


metadata_path = dataset.metadata_files[0]
import json
with open(metadata_path,'r') as f:
    meta = json.load(f)

gt_positions = []
for s in meta["sources"]:
    gt_positions.append([s["azimuth_deg"], s["elevation_deg"]])
gt_positions = np.array(gt_positions)
print("\nGround truth positions (from metadata):")
print(gt_positions)


matches, missed, false_alarms, correct = match_sources(
    gt_positions,
    np.array(pred_pos),
    max_error=30   # 匹配角度阈值
)

print("\n===== MATCH RESULTS =====")
print("Matches:", matches)
print("Missed detections:", missed)
print("False alarms:", false_alarms)
print("Correct matches:", correct)
heatmap_plot(heatmap, title="Ground Truth Heatmap")