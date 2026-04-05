import os
import queue
import time
from datetime import datetime, timezone

from clock import get_utc_now

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000  # 16 kHz mono — optimal for Whisper
CHANNELS = 1


class Recorder:
    def __init__(self, save_audio_path=None):
        self._q = queue.Queue()
        self._save_path = save_audio_path
        self._start_time = None
        self._stop_time = None
        self._start_utc = None
        self._stream = None
        self._wav_writer = None
        # Rolling buffer for overlap: holds the last N samples from the previous chunk
        self._overlap_audio = np.zeros(0, dtype="float32")
        # Wall-clock time of the previous drain_chunk call; used to compute time_offset
        self._last_drain_wall: float = None

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata.copy())

    @property
    def start_utc(self) -> datetime:
        return self._start_utc

    def start(self):
        self._overlap_audio = np.zeros(0, dtype="float32")
        self._last_drain_wall = None
        self._start_time = time.time()
        self._start_utc = get_utc_now()
        if self._save_path:
            self._wav_writer = sf.SoundFile(
                self._save_path, mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16"
            )
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
        end = self._stop_time if self._stop_time is not None else time.time()
        return int(end - self._start_time)

    def stop_stream(self) -> None:
        """Stop the audio input stream without draining the queue.

        Call this to freeze the recording length before waiting on other work
        (e.g. a transcription thread).  The queue may still hold unread audio;
        call stop() afterwards to drain it.
        """
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._stop_time is None:
            self._stop_time = time.time()

    def drain_chunk(self, overlap_samples: int = 0) -> tuple:
        """Drain all audio currently in the queue and return it as a flat float32 array.

        The returned array is prefixed with *overlap_samples* samples carried over
        from the end of the previous chunk.  This ensures words that straddle a
        chunk boundary are heard in full by Whisper.

        Also writes the new (non-overlap) audio to .wav incrementally if enabled.

        Returns
        -------
        (audio, time_offset, actual_overlap_seconds) where:
          - audio is the combined (overlap + new) float32 array
          - time_offset is the session-elapsed seconds at the START of the audio
          - actual_overlap_seconds is how many seconds of overlap were prepended
            (0.0 for the first call when there is no previous chunk)
        """
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get_nowait())

        now_wall = time.time()
        new_audio = np.concatenate(chunks, axis=0).flatten() if chunks else np.zeros(0, dtype="float32")

        if self._wav_writer is not None and len(new_audio) > 0:
            self._wav_writer.write(new_audio)

        # Actual overlap that will be prepended (length of stored overlap buffer)
        actual_overlap_secs = len(self._overlap_audio) / SAMPLE_RATE

        # time_offset = session time at the START of the overlap prefix.
        # The overlap buffer was captured ending at _last_drain_wall, so it
        # starts at _last_drain_wall - actual_overlap_secs (relative to _start_time).
        # On the first call there is no previous drain, so time_offset = 0.
        if self._last_drain_wall is not None:
            time_offset = max(0.0, (self._last_drain_wall - self._start_time) - actual_overlap_secs)
        else:
            time_offset = 0.0

        self._last_drain_wall = now_wall

        # Build chunk: overlap tail from previous + new audio
        combined = np.concatenate([self._overlap_audio, new_audio]) if len(self._overlap_audio) > 0 else new_audio

        # Store the new overlap tail for the next call
        if overlap_samples > 0 and len(new_audio) >= overlap_samples:
            self._overlap_audio = new_audio[-overlap_samples:]
        else:
            self._overlap_audio = np.zeros(0, dtype="float32")

        return combined, time_offset, actual_overlap_secs

    def stop(self) -> tuple:
        """Stop the stream (if still running) and drain remaining audio.

        Returns (final_audio_ndarray, sample_rate, time_offset).
        Call stop_stream() first if you need to freeze the recording length
        before waiting on other work.
        """
        self.stop_stream()  # no-op if already called

        remaining, time_offset, _ = self.drain_chunk(overlap_samples=0)

        if self._wav_writer is not None:
            self._wav_writer.close()
            self._wav_writer = None
            if self._save_path:
                print(f"  Audio saved: {os.path.basename(self._save_path)}")

        return remaining, SAMPLE_RATE, time_offset

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
