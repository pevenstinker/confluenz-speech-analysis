import os
import queue
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000  # 16 kHz mono — optimal for Whisper
CHANNELS = 1


class Recorder:
    def __init__(self, save_audio_path=None):
        self._q = queue.Queue()
        self._chunks = []
        self._stream = None
        self._save_path = save_audio_path
        self._start_time = None

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata.copy())

    def start(self):
        self._chunks = []
        self._start_time = time.time()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def get_elapsed(self) -> int:
        if self._start_time is None:
            return 0
        return int(time.time() - self._start_time)

    def stop(self):
        """Stop the stream and return (audio_ndarray, sample_rate)."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
        # drain any remaining frames from the queue
        while not self._q.empty():
            self._chunks.append(self._q.get_nowait())

        audio = np.concatenate(self._chunks, axis=0).flatten() if self._chunks else np.zeros(0, dtype="float32")

        if self._save_path and len(audio) > 0:
            sf.write(self._save_path, audio, SAMPLE_RATE)
            print(f"  Audio saved: {os.path.basename(self._save_path)}")

        return audio, SAMPLE_RATE

    @staticmethod
    def list_devices():
        return sd.query_devices()

    @staticmethod
    def default_input_name() -> str:
        try:
            dev = sd.query_devices(kind="input")
            return dev["name"]
        except Exception:
            return "unknown"
