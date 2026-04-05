"""
Re-classify an existing results JSON file (useful for testing / re-running Phase 2
without having to re-record).

Usage:
    python classify_file.py results-2026-04-01-dialogue-42.json
    python classify_file.py results-2026-04-01-dialogue-42.json --output reclassified.json
"""
import json
import sys

import click

from classifier import classify_segments


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", default=None, help="Output path (default: overwrites input file).")
def classify_file(input_file, output):
    """Classify (or re-classify) all segments in a results JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("  No segments found in file.")
        sys.exit(1)

    classifier_backend, classifier_model = classify_segments(segments)

    inquiry_segs = [s for s in segments if s["label"] == "inquiry"]
    advocacy_segs = [s for s in segments if s["label"] == "advocacy"]
    inquiry_dur = sum(s["end"] - s["start"] for s in inquiry_segs)
    advocacy_dur = sum(s["end"] - s["start"] for s in advocacy_segs)
    total_seg_dur = inquiry_dur + advocacy_dur

    stats = data.setdefault("stats", {})
    stats["classifierBackend"] = classifier_backend
    stats["classifierModel"] = classifier_model
    stats["labelledSegments"] = len(segments)
    stats["inquirySegments"] = len(inquiry_segs)
    stats["advocacySegments"] = len(advocacy_segs)
    stats["inquiryDurationSeconds"] = round(inquiry_dur, 1)
    stats["advocacyDurationSeconds"] = round(advocacy_dur, 1)
    stats["inquiryDurationPercent"] = round(inquiry_dur / total_seg_dur * 100, 1) if total_seg_dur > 0 else 0.0

    out_path = output or input_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  {len(segments)} segments classified.")
    print(f"  Inquiry:  {len(inquiry_segs)} segments ({stats['inquiryDurationPercent']}% of speech time)")
    print(f"  Advocacy: {len(advocacy_segs)} segments")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    classify_file()
