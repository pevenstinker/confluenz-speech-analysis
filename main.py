"""
confluenz-speech-analysis — Phase 2: Transcription + Advocacy/Inquiry Classification
Usage:  python main.py [OPTIONS]
"""
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import click

from classifier import classify_segments
from config import config
from recorder import Recorder
from transcriber import transcribe

BANNER = """
  ___           __ _
 / __|___ _ _  / _| |___ _  _ ___ _ _  ____
| (__/ _ \\ ' \\|  _| / -_) || / -_) ' \\|_  /
 \\___\\___/_||_|_| |_\\___|\\_,_\\___|_||_/__/

  Speech Analysis  —  Phase 1: Transcription
"""


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

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    id_part = f"-dialogue-{dialogue_id}" if dialogue_id is not None else ""
    base_name = f"{date_str}{id_part}"

    audio_path = os.path.join(os.getcwd(), f"audio-{base_name}.wav") if config.save_audio else None
    transcript_path = os.path.join(os.getcwd(), f"transcript-{base_name}.txt") if config.save_transcript_txt else None
    json_path = output or os.path.join(os.getcwd(), f"results-{base_name}.json")

    print(f"  Model          : {config.whisper_model}")
    print(f"  Classifier     : {config.classifier} ({config.ollama_model if config.classifier == 'ollama' else 'built-in rules'})")
    print(f"  Save audio     : {'yes → ' + os.path.basename(audio_path) if audio_path else 'no'}")
    print(f"  Save transcript: {'yes → ' + os.path.basename(transcript_path) if transcript_path else 'no'}")
    print(f"  Text in JSON   : {'yes' if config.save_text_in_json else 'NO — privacy mode'}")
    print(f"  Input device   : {Recorder.default_input_name()}")
    print()

    recorder = Recorder(save_audio_path=audio_path)
    recorder.start()

    stop_event = threading.Event()
    spinner = threading.Thread(target=_elapsed_spinner, args=(recorder, stop_event), daemon=True)
    spinner.start()

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    stop_event.set()
    print()  # newline after the in-place elapsed line
    print("  Stopping...")
    audio, sample_rate = recorder.stop()

    duration = len(audio) / sample_rate
    if duration < 0.5:
        print("  No audio captured. Exiting.")
        sys.exit(1)

    print(f"  Captured {duration:.1f}s of audio.")
    print(f"  Transcribing with faster-whisper ({config.whisper_model})...")
    print("  (This may take a minute or two on CPU — roughly 1/10 of audio length)")
    print()

    segments, model_version = transcribe(
        audio,
        sample_rate,
        model_size=config.whisper_model,
        save_transcript_path=transcript_path,
    )

    print("  Classifying segments (advocacy / inquiry)...")
    classifier_backend, classifier_model = classify_segments(segments)

    languages_seen = sorted({s["language"] for s in segments if s["language"]})

    inquiry_segs = [s for s in segments if s["label"] == "inquiry"]
    advocacy_segs = [s for s in segments if s["label"] == "advocacy"]
    inquiry_dur = sum(s["end"] - s["start"] for s in inquiry_segs)
    advocacy_dur = sum(s["end"] - s["start"] for s in advocacy_segs)
    total_seg_dur = inquiry_dur + advocacy_dur

    if not config.save_text_in_json:
        for seg in segments:
            seg.pop("text", None)

    result = {
        "dialogueId": dialogue_id,
        "recordedAt": now.isoformat(),
        "modelVersion": model_version,
        "segments": segments,
        "stats": {
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
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print(f"  Done — {len(segments)} segments transcribed and classified.")
    print(f"  Inquiry:  {len(inquiry_segs)} segments ({result['stats']['inquiryDurationPercent']}% of speech time)")
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
