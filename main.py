"""
confluenz-speech-analysis — Phase 2: Transcription + Advocacy/Inquiry Classification
Usage:  python main.py [OPTIONS]
"""
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import click

from classifier import classify_segments
from clock import get_utc_now
from config import config
from recorder import Recorder, SAMPLE_RATE
from transcriber import transcribe, transcribe_chunk

BANNER = """
  ___           __ _
 / __|___ _ _  / _| |___ _  _ ___ _ _  ____
| (__/ _ \\ ' \\|  _| / -_) || / -_) ' \\|_  /
 \\___\\___/_||_|_| |_\\___|\\_,_\\___|_||_/__/

  Speech Analysis  —  Phase 1: Transcription
"""

SCHEMA_VERSION = "2.0"


def _fmt(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _elapsed_spinner(recorder: Recorder, stop_event: threading.Event):
    while not stop_event.is_set():
        elapsed = _fmt(recorder.get_elapsed())
        print(f"\r  Recording... {elapsed}  (press ENTER to stop)", end="", flush=True)
        time.sleep(1)


def _add_utc_times(segments: list, recording_start_utc: datetime) -> None:
    """Add startTime / endTime ISO strings to each segment in-place."""
    for seg in segments:
        seg["startTime"] = (recording_start_utc + timedelta(seconds=seg["start"])).isoformat()
        seg["endTime"] = (recording_start_utc + timedelta(seconds=seg["end"])).isoformat()


def _compute_stats(segments: list, duration: float, classifier_backend: str, classifier_model: str) -> dict:
    languages_seen = sorted({s["language"] for s in segments if s.get("language")})
    inquiry_segs = [s for s in segments if s["label"] == "inquiry"]
    advocacy_segs = [s for s in segments if s["label"] == "advocacy"]
    inquiry_dur = sum(s["end"] - s["start"] for s in inquiry_segs)
    advocacy_dur = sum(s["end"] - s["start"] for s in advocacy_segs)
    total_seg_dur = inquiry_dur + advocacy_dur
    return {
        "totalSegments": len(segments),
        "audioDurationSeconds": round(duration, 1),
        "detectedLanguages": languages_seen,
        "classifierBackend": classifier_backend,
        "classifierModel": classifier_model,
        "labelledSegments": len(segments),
        "inquirySegments": len(inquiry_segs),
        "advocacySegments": len(advocacy_segs),
        "inquiryDurationSeconds": round(inquiry_dur, 1),
        "advocacyDurationSeconds": round(advocacy_dur, 1),
        "inquiryDurationPercent": round(inquiry_dur / total_seg_dur * 100, 1) if total_seg_dur > 0 else 0.0,
    }


def _write_json(json_path: str, result: dict) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def _append_transcript(transcript_path: str, segments: list) -> None:
    with open(transcript_path, "a", encoding="utf-8") as f:
        for seg in segments:
            h = int(seg["start"] // 3600)
            m = int((seg["start"] % 3600) // 60)
            s = int(seg["start"] % 60)
            f.write(f"[{h:02d}:{m:02d}:{s:02d}] {seg['text']}\n")


@click.command()
@click.option("--dialogue-id", default=None, type=int, help="Confluenz dialogue ID (used in output filename).")
@click.option("--model", default=None, help="Whisper model size: tiny / base / small / medium. Overrides .env.")
@click.option("--save-audio", is_flag=True, default=False, help="Save recorded audio as .wav. Overrides .env.")
@click.option("--save-transcript-txt", "--save-transcript", is_flag=True, default=False, help="Save plain-text .txt transcript file. Overrides .env.")
@click.option("--no-text-in-json", is_flag=True, default=False, help="Omit transcribed text from the results JSON (privacy mode). Overrides .env.")
@click.option("--output", default=None, help="Output JSON filename (default: auto-generated).")
def record(dialogue_id, model, save_audio, save_transcript_txt, no_text_in_json, output):
    """Record microphone audio, transcribe with Whisper, and save a JSON result."""
    print(BANNER)

    # CLI flags override .env (only when explicitly passed)
    config.apply_overrides(
        model=model,
        save_audio=save_audio if save_audio else None,
        save_transcript_txt=save_transcript_txt if save_transcript_txt else None,
        save_text_in_json=False if no_text_in_json else None,
    )

    now = get_utc_now()
    date_str = now.strftime("%Y-%m-%d")
    id_part = f"-dialogue-{dialogue_id}" if dialogue_id is not None else ""
    base_name = f"{date_str}{id_part}"

    audio_path = os.path.join(os.getcwd(), f"audio-{base_name}.wav") if config.save_audio else None
    transcript_path = os.path.join(os.getcwd(), f"transcript-{base_name}.txt") if config.save_transcript_txt else None
    json_path = output or os.path.join(os.getcwd(), f"results-{base_name}.json")

    model_version = f"faster-whisper-{config.whisper_model}-v3"

    chunk_interval = config.chunk_interval
    overlap_seconds = config.chunk_overlap
    overlap_samples = int(overlap_seconds * SAMPLE_RATE)

    print(f"  Model          : {config.whisper_model}")
    print(f"  Classifier     : {config.classifier} ({config.ollama_model if config.classifier == 'ollama' else 'built-in rules'})")
    print(f"  Chunk interval : {chunk_interval}s  (overlap: {overlap_seconds}s)")
    print(f"  Save audio     : {'yes → ' + os.path.basename(audio_path) if audio_path else 'no'}")
    print(f"  Save transcript: {'yes → ' + os.path.basename(transcript_path) if transcript_path else 'no'}")
    print(f"  Text in JSON   : {'yes' if config.save_text_in_json else 'NO — privacy mode'}")
    print(f"  Input device   : {Recorder.default_input_name()}")
    print()

    recorder = Recorder(save_audio_path=audio_path)
    recorder.start()
    recording_start_utc = recorder.start_utc

    # Seed the in-progress JSON immediately so the file exists from the start
    all_segments: list = []
    classifier_backend = config.classifier
    classifier_model = config.ollama_model if config.classifier == "ollama" else "regex"
    prev_segment = None  # last classified segment for cross-chunk context

    result = {
        "schemaVersion": SCHEMA_VERSION,
        "dialogueId": dialogue_id,
        "recordedAt": now.isoformat(),
        "modelVersion": model_version,
        "segments": [],
        "stats": _compute_stats([], 0.0, classifier_backend, classifier_model),
    }
    _write_json(json_path, result)

    # --- chunk processing thread -------------------------------------------
    chunk_stop = threading.Event()
    chunk_error: list = []  # used to surface exceptions from the thread

    def _chunk_loop():
        nonlocal prev_segment, classifier_backend, classifier_model
        chunk_index = 0
        while not chunk_stop.is_set():
            chunk_stop.wait(timeout=chunk_interval)
            chunk_index += 1

            audio_chunk, time_offset, actual_overlap_secs = recorder.drain_chunk(overlap_samples=overlap_samples)
            if len(audio_chunk) < SAMPLE_RATE * 0.5:
                continue  # too short / silent — skip this window

            try:
                new_segs = transcribe_chunk(
                    audio_chunk,
                    SAMPLE_RATE,
                    model_size=config.whisper_model,
                    time_offset=time_offset,
                    overlap_seconds=actual_overlap_secs,
                )
            except Exception as e:
                chunk_error.append(e)
                return

            if not new_segs:
                continue

            try:
                cb, cm = classify_segments(new_segs, prev_segment=prev_segment)
                classifier_backend = cb
                classifier_model = cm
            except Exception as e:
                chunk_error.append(e)
                return

            prev_segment = new_segs[-1]
            _add_utc_times(new_segs, recording_start_utc)

            if transcript_path:
                _append_transcript(transcript_path, new_segs)

            # Strip text before appending to the running list if privacy mode
            if not config.save_text_in_json:
                for seg in new_segs:
                    seg.pop("text", None)

            all_segments.extend(new_segs)

            total_dur = recorder.get_elapsed()
            result["segments"] = all_segments
            result["stats"] = _compute_stats(all_segments, total_dur, classifier_backend, classifier_model)
            _write_json(json_path, result)

            seg_count = len(all_segments)
            print(f"\r  Chunk {chunk_index} processed — {seg_count} segments so far.  ", end="", flush=True)

    chunk_thread = threading.Thread(target=_chunk_loop, daemon=True)
    chunk_thread.start()

    # --- spinner (same thread as before, runs until ENTER) -----------------
    stop_event = threading.Event()
    spinner = threading.Thread(target=_elapsed_spinner, args=(recorder, stop_event), daemon=True)
    spinner.start()

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    stop_event.set()
    print()

    if config.stop_delay > 0:
        print(f"  Finishing... ({config.stop_delay}s)", flush=True)
        time.sleep(config.stop_delay)

    # Stop the audio stream NOW so the recording length is fixed and no new
    # audio accumulates while we wait for an in-progress transcription to finish.
    recorder.stop_stream()
    chunk_stop.set()
    print("  Stopping...")

    # Wait for any in-progress chunk to finish BEFORE draining the recorder,
    # so the chunk thread and the final-audio path don't race on the queue.
    chunk_thread.join(timeout=120)

    # Stop recording and collect any audio that arrived after the last chunk drain
    final_audio, _, final_time_offset = recorder.stop()

    if chunk_error:
        print(f"  Warning: chunk processing error — {chunk_error[0]}")

    duration = recorder.get_elapsed()

    # Process final partial chunk (audio drained after stop, no overlap)
    if len(final_audio) >= SAMPLE_RATE * 0.5:
        print("  Processing final audio chunk...")
        final_segs = transcribe_chunk(
            final_audio,
            SAMPLE_RATE,
            model_size=config.whisper_model,
            time_offset=final_time_offset,
            overlap_seconds=0.0,
        )
        if final_segs:
            classify_segments(final_segs, prev_segment=prev_segment)
            _add_utc_times(final_segs, recording_start_utc)
            if transcript_path:
                _append_transcript(transcript_path, final_segs)
            if not config.save_text_in_json:
                for seg in final_segs:
                    seg.pop("text", None)
            all_segments.extend(final_segs)

    # Write final result
    result["segments"] = all_segments
    result["stats"] = _compute_stats(all_segments, float(duration), classifier_backend, classifier_model)
    _write_json(json_path, result)

    stats = result["stats"]
    inquiry_segs = [s for s in all_segments if s["label"] == "inquiry"]
    advocacy_segs = [s for s in all_segments if s["label"] == "advocacy"]

    print()
    print(f"  Done — {len(all_segments)} segments transcribed and classified.")
    print(f"  Inquiry:  {len(inquiry_segs)} segments ({stats['inquiryDurationPercent']}% of speech time)")
    print(f"  Advocacy: {len(advocacy_segs)} segments")
    print()
    print(f"  Saved: {os.path.basename(json_path)}")
    if transcript_path:
        print(f"  Saved: {os.path.basename(transcript_path)}")
    if audio_path:
        print(f"  Saved: {os.path.basename(audio_path)}")
    print()


if __name__ == "__main__":
    record()
