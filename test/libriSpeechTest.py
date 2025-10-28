import os

import librosa
import soundfile as sf

# 读入音频
# dir_path = "/home/kehan.zeng/DATA2/librispeech/LibriSpeech/test-clean/121/121726"
dir_path = "/home/kehan.zeng/DATA2/voice/mssl_libri/wavs/sample_11"
for file in os.listdir(dir_path):
    if file.endswith(".wav"):
        file_path = os.path.join(dir_path, file)
        audio, fs = sf.read(file_path)
        print(f"采样率: {fs}, 音频长度(秒): {len(audio)}")
        print(f"音频数据类型: {audio.dtype}, 音频最大值: {max(audio)}, 音频最小值: {min(audio)}")
