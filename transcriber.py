import os
import tempfile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

_model_cache: dict = {}


def _get_model(model_size: str) -> WhisperModel:
    if model_size not in _model_cache:
        print(f'  Loading Whisper model "{model_size}" (first run downloads weights ~500 MB)...')
        _model_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("  Model ready.")
    return _model_cache[model_size]


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    model_size: str,
    save_transcript_path: str = None,
):
    """Transcribe *audio* and return (segments_list, model_version_str).

    Parameters
    ----------
    audio:                 float32 numpy array, mono
    sample_rate:           should be 16 000
    model_size:            e.g. "small"
    save_transcript_path:  if set, write a timestamped plain-text transcript here
    """
    model = _get_model(model_size)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, audio, sample_rate)
        tmp.close()
        segments_iter, info = model.transcribe(tmp.name, language=None, vad_filter=True)
        segments = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "language": info.language,
                    "label": None,
                    "confidence": None,
                }
            )
    finally:
        os.unlink(tmp.name)

    model_version = f"faster-whisper-{model_size}-v3"

    if save_transcript_path and segments:
        with open(save_transcript_path, "w", encoding="utf-8") as f:
            for seg in segments:
                h = int(seg["start"] // 3600)
                m = int((seg["start"] % 3600) // 60)
                s = int(seg["start"] % 60)
                f.write(f"[{h:02d}:{m:02d}:{s:02d}] {seg['text']}\n")
        print(f"  Transcript saved: {os.path.basename(save_transcript_path)}")

    return segments, model_version
