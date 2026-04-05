"""
Evaluate a classified synthetic dialogue against its groundTruth labels.

Usage:
    python eval_synthetic.py <classified_json>
    python eval_synthetic.py dialogue-ai-classified.json
"""
import json
import sys


def evaluate(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    results = []
    for seg in segments:
        gt = seg.get("groundTruth")
        label = seg.get("label")
        if gt is None or label is None:
            continue
        correct = label == gt
        results.append({
            "text": seg.get("text", "")[:80],
            "speaker": seg.get("speaker", "?"),
            "groundTruth": gt,
            "label": label,
            "correct": correct,
        })

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    wrong = [r for r in results if not r["correct"]]

    print(f"\n{'='*60}")
    print(f"  File: {path}")
    print(f"  Correct: {correct_count}/{total}   Accuracy: {correct_count/total*100:.1f}%")
    print(f"{'='*60}")

    if wrong:
        print(f"\nMISCLASSIFIED ({len(wrong)}):")
        for r in wrong:
            print(f"  [{r['speaker']}] GT={r['groundTruth']:<10} Got={r['label']:<10}  \"{r['text']}...\"")
    else:
        print("\n  All segments correct!")

    print()
    return correct_count, total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_synthetic.py <classified_json> [<classified_json2> ...]")
        sys.exit(1)
    totals = [evaluate(p) for p in sys.argv[1:]]
    if len(totals) > 1:
        grand_correct = sum(c for c, _ in totals)
        grand_total = sum(t for _, t in totals)
        print(f"GRAND TOTAL: {grand_correct}/{grand_total}  ({grand_correct/grand_total*100:.1f}%)")
