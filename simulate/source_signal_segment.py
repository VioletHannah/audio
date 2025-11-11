"""
文件: slice_audio_to_frames.py
用途: 将目录中每个音频文件切成长度为16000、50%重叠(跳步8000)的帧，采样率为48000Hz，保存为 wav 与 metadata json。
用法示例:
    python slice_audio_to_frames.py --input_dir /path/to/source --output_dir /path/to/frames --pad discard
"""
import os
import argparse
import json
import math
import glob

import numpy as np
import soundfile as sf
from scipy.signal import resample


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_mono_and_resample(path, target_fs):
    data, orig_fs = sf.read(path, always_2d=False)
    # 转为单声道
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    if orig_fs != target_fs and len(data) > 0:
        new_len = int(round(len(data) * float(target_fs) / float(orig_fs)))
        if new_len <= 0:
            return np.zeros(0, dtype=np.float32), target_fs
        data = resample(data, new_len).astype(np.float32)
    return data, target_fs


def frames_from_signal(signal, frame_len=16000, hop=8000, pad_method='discard'):
    """
    返回 list of numpy arrays，每个长度为 frame_len。
    pad_method: 'zero' or 'repeat' or 'discard'
    当 pad_method == 'discard' 时：若最后一帧不足 frame_len 则舍弃；若整个信号小于等于 frame_len 也舍弃（返回空列表）。
    """
    n = len(signal)
    if n == 0:
        # 空信号：如果 discard 则不返回帧，否则返回单个全零帧
        if pad_method == 'discard':
            return []
        return [np.zeros(frame_len, dtype=np.float64)]

    if n <= frame_len:
        if pad_method == 'repeat' and n > 0:
            reps = int(math.ceil(frame_len / n))
            extended = np.tile(signal, reps)[:frame_len]
            return [extended.astype(np.float64)]
        elif pad_method == 'discard':
            return []
        else:
            buf = np.zeros(frame_len, dtype=np.float64)
            buf[:n] = signal
            return [buf]

    frames = []
    num_frames = int(math.ceil((n - frame_len) / float(hop))) + 1
    for i in range(num_frames):
        start = i * hop
        end = start + frame_len
        if end <= n:
            frames.append(signal[start:end].astype(np.float64))
        else:
            # 末尾不足一帧
            remain = n - start
            if pad_method == 'repeat' and remain > 0:
                tail = np.tile(signal[start:], int(math.ceil(frame_len / remain)))[:frame_len]
                frames.append(tail.astype(np.float64))
            elif pad_method == 'discard':
                # 舍弃最后不足一帧的片段（不加入，也不继续）
                break
            else:
                buf = np.zeros(frame_len, dtype=np.float64)
                if remain > 0:
                    buf[:remain] = signal[start:]
                frames.append(buf)
    return frames


def process_directory(input_dir, output_dir, exts=('.wav', '.flac', '.mp3'), frame_len=16000, fs=48000, overlap=0.5, pad='zero'):
    hop = int(frame_len * (1.0 - overlap))
    file_list = []
    for ext in exts:
        file_list.extend(glob.glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
    if not file_list:
        raise ValueError(f"未在 {input_dir} 下找到支持的音频文件")

    metadata = {}
    for filepath in file_list:
        rel_path = os.path.relpath(filepath, input_dir)
        name = os.path.splitext(rel_path.replace(os.sep, '_'))[0]
        out_subdir = os.path.join(output_dir, name)
        ensure_dir(out_subdir)

        signal, _ = load_mono_and_resample(filepath, fs)
        frames = frames_from_signal(signal, frame_len=frame_len, hop=hop, pad_method=pad)

        frame_infos = []
        for idx, frame in enumerate(frames):
            out_name = f"{name}_frame_{idx:05d}.wav"
            out_path = os.path.join(out_subdir, out_name)
            sf.write(out_path, frame, fs, subtype='PCM_16')
            frame_infos.append({
                'frame_index': idx,
                'file': os.path.relpath(out_path, output_dir),
                'start_sample': int(idx * hop),
                'frame_len': frame_len
            })

        metadata[rel_path] = {
            'num_frames': len(frames),
            'frames': frame_infos
        }

    # 保存全局 metadata
    ensure_dir(output_dir)
    meta_path = os.path.join(output_dir, 'slices_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"处理完成: {len(file_list)} 个文件 -> 切片保存在 {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description='切分音频为重叠帧 (16000 samples, 48kHz, 50% overlap)')
    p.add_argument('--input_dir', required=True, help='输入声源库目录 (递归搜索音频文件)')
    p.add_argument('--output_dir', required=True, help='输出切片目录')
    p.add_argument('--frame_len', type=int, default=16000, help='帧长度 (samples)')
    p.add_argument('--fs', type=int, default=48000, help='目标采样率')
    p.add_argument('--overlap', type=float, default=0.5, help='重叠比例 (0-1)')
    p.add_argument('--pad', choices=['zero', 'repeat', 'discard'], default='zero', help='尾段处理方法: zero、repeat 或 discard（舍弃不足一帧）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frame_len=args.frame_len,
        fs=args.fs,
        overlap=args.overlap,
        pad=args.pad
    )