"""
Interactive segment labeller — builds training data for the Phase 2 classifier.

For each segment, shows the current (AI) label and asks you to confirm or correct.
Confirmed labels are appended to training-data.jsonl.

Usage:
    python label_file.py results-2026-04-01-dialogue-44.json
    python label_file.py results-*.json          # multiple files
    python label_file.py results-44.json --only-uncertain   # skip confident ones

Controls:
    ENTER   — accept the current AI label
    a       — mark as advocacy
    i       — mark as inquiry
    s       — skip (don't add to training data)
    q       — quit (saves progress so far)
"""
import json
import os
import sys

import click

TRAINING_FILE = os.path.join(os.path.dirname(__file__), "training-data.jsonl")

ADV = "advocacy"
INQ = "inquiry"

_LABEL_COLOURS = {ADV: "\033[93m", INQ: "\033[96m"}  # yellow / cyan
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_RED = "\033[91m"


def _colour(label):
    return f"{_LABEL_COLOURS.get(label, '')}{label.upper()}{_RESET}"


def _load_existing_training():
    """Return set of texts already in training-data.jsonl (to avoid duplicates)."""
    seen = set()
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)["text"])
                    except Exception:
                        pass
    return seen


def _append_training(text, label, language):
    with open(TRAINING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label, "language": language}, ensure_ascii=False) + "\n")


def _count_training():
    if not os.path.exists(TRAINING_FILE):
        return 0
    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@click.command()
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--only-uncertain", is_flag=True, default=False,
              help="Only show segments where confidence < 0.9 (skip clear-cut ones).")
@click.option("--update-json", is_flag=True, default=False,
              help="Also write corrected labels back into the results JSON file.")
def label_file(input_files, only_uncertain, update_json):
    """Review AI labels segment by segment and build a training dataset."""

    existing_texts = _load_existing_training()

    total_added = 0
    total_corrected = 0

    for filepath in input_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            print(f"  No segments in {filepath}, skipping.")
            continue

        dialogue_id = data.get("dialogueId", "?")
        print(f"\n{_BOLD}── {os.path.basename(filepath)}  (dialogue {dialogue_id}, {len(segments)} segments) ──{_RESET}\n")

        changed = False

        for idx, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                print(f"  {_DIM}[{idx+1}/{len(segments)}] No text (privacy mode) — skipping.{_RESET}")
                continue
            language = seg.get("language", "en")
            ai_label = seg.get("label") or "?"
            confidence = seg.get("confidence") or 0.0

            if only_uncertain and confidence >= 0.9:
                continue

            if text in existing_texts:
                # already labelled — still apply to JSON if needed but don't re-prompt
                continue

            pct = int(confidence * 100)
            conf_display = f"{_GREEN}{pct}%{_RESET}" if pct >= 80 else f"{_RED}{pct}%{_RESET}"

            print(f"  {_DIM}[{idx+1}/{len(segments)}]{_RESET}  {_colour(ai_label)} {conf_display}")
            print(f"  {_BOLD}{text}{_RESET}")
            print()

            while True:
                raw = input("  ENTER=accept  a=advocacy  i=inquiry  s=skip  q=quit  > ").strip().lower()

                if raw == "q":
                    print(f"\n  Labelling stopped. {total_added} entries added ({total_corrected} corrections).")
                    print(f"  Training file: {TRAINING_FILE}  ({_count_training()} total examples)\n")
                    if update_json and changed:
                        _save_json(filepath, data)
                    sys.exit(0)

                elif raw == "s":
                    print(f"  {_DIM}Skipped.{_RESET}\n")
                    break

                elif raw in ("", "a", "i"):
                    if raw == "":
                        final_label = ai_label
                    elif raw == "a":
                        final_label = ADV
                    else:
                        final_label = INQ

                    is_correction = final_label != ai_label
                    marker = f"{_RED}✗ corrected to {final_label.upper()}{_RESET}" if is_correction else f"{_GREEN}✓ confirmed{_RESET}"
                    print(f"  {marker}\n")

                    _append_training(text, final_label, language)
                    existing_texts.add(text)
                    total_added += 1
                    if is_correction:
                        total_corrected += 1
                        seg["label"] = final_label
                        changed = True

                    break
                else:
                    print("  Invalid key. Use ENTER, a, i, s, or q.")

        if update_json and changed:
            _save_json(filepath, data)

    print(f"\n  Done. {total_added} entries added ({total_corrected} corrections).")
    print(f"  Training file: {TRAINING_FILE}")
    n = _count_training()
    print(f"  Total examples: {n}")
    if n < 100:
        print(f"  ({100 - n} more needed before training a classifier — keep labelling!)")
    elif n < 200:
        print(f"  Getting there! {200 - n} more for a robust model.")
    else:
        print(f"  You have enough data to train. Run: .\\run train.py")
    print()


def _save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  {_GREEN}JSON updated: {os.path.basename(filepath)}{_RESET}")


if __name__ == "__main__":
    label_file()
