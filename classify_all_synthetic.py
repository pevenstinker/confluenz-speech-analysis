"""Run classify_file on all 5 new synthetic dialogues in sequence."""
import subprocess
import sys

files = [
    ("synthetic-vaccines-dialogue.json", "synthetic-vaccines-dialogue-classified.json"),
    ("synthetic-gun-rights-dialogue.json", "synthetic-gun-rights-dialogue-classified.json"),
    ("synthetic-avengers-dialogue.json", "synthetic-avengers-dialogue-classified.json"),
    ("synthetic-jared-leto-dialogue.json", "synthetic-jared-leto-dialogue-classified.json"),
    ("synthetic-religion-dialogue.json", "synthetic-religion-dialogue-classified.json"),
]

python = sys.executable

for infile, outfile in files:
    print(f"\n{'='*60}")
    print(f"  Classifying: {infile}")
    print(f"{'='*60}")
    result = subprocess.run(
        [python, "classify_file.py", infile, "--output", outfile],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")

print("\nAll done.")
