# python
import soundfile as sf

_SUBTYPE_TO_BITS = {
    'PCM_U8': 8,
    'PCM_S8': 8,
    'PCM_16': 16,
    'PCM_24': 24,
    'PCM_32': 32,
    'PCM_64': 64,
    'FLOAT': 32,    # 32-bit float
    'DOUBLE': 64,   # 64-bit float (if supported)
    # 常见非 PCM 子类型：不定长或无明确“位深”
    'ULAW': None,
    'ALAW': None
}

def inspect_audio_precision(path):
    """
    返回 dict:
      { 'path', 'format', 'subtype', 'samplerate', 'channels', 'frames', 'duration',
        'bit_depth' (int or None), 'is_float' (bool or None) }
    bit_depth 为 None 表示无法确定（如压缩格式或未知子类型）。
    """
    info = sf.info(path)
    subtype = info.subtype
    fmt = info.format
    bit_depth = _SUBTYPE_TO_BITS.get(subtype, None)
    is_float = None
    if subtype in ('FLOAT', 'DOUBLE'):
        is_float = True
    elif subtype is not None:
        # 若 bit_depth 可知且不是 FLOAT/DOUBLE 则视为整数 PCM
        is_float = False if bit_depth is not None else None

    return {
        'path': path,
        'format': fmt,
        'subtype': subtype,
        'samplerate': info.samplerate,
        'channels': info.channels,
        'frames': info.frames,
        'duration': None if info.frames is None else float(info.frames) / info.samplerate,
        'bit_depth': bit_depth,
        'is_float': is_float
    }

if __name__ == '__main__':
    p = r'/home/kehan.zeng/DATA2/voice/bal_train_segment/6z9Omn1fhjo_segment0026.flac'
    q = r'/home/kehan.zeng/DATA2/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac'
    info = inspect_audio_precision(q)
    print(info)