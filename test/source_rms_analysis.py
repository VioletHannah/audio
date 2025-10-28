"""
分析一下音频数据集中所有音频 RMS值的分布情况
"""
import glob
import numpy as np
import os
import soundfile as sf
import json
import matplotlib.pyplot as plt

def analyze_rms_distribution(dataset_path):
    audio_files = glob.glob(os.path.join(dataset_path, "*.flac"))
    rms_values = []
    for audio_file in audio_files:
        if len(rms_values) % 50 == 0:
            print(f"Processing file {len(rms_values)+1}/{len(audio_files)}")
        try:
            audio, fs = sf.read(audio_file)
            rms = np.sqrt(np.mean(audio**2))
            print(f"{os.path.basename(audio_file)}: RMS = {rms:.6f}")
            rms_values.append(rms)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    rms_values = np.array(rms_values)
    print(f"Processed {len(rms_values)} audio files.")
    print(f"RMS Mean: {np.mean(rms_values):.6f}, RMS Std: {np.std(rms_values):.6f}")
    # 绘制RMS分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(rms_values, bins=50, color='blue', alpha=0.7)
    plt.title("RMS Distribution")
    plt.xlabel("RMS Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    dataset_path = "/home/kehan.zeng/DATA2/voice/bal_train_segment"
    analyze_rms_distribution(dataset_path)
