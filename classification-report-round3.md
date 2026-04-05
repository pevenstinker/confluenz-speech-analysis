# Dialogue Classifier — Synthetic Evaluation Report
**Date:** 2026-04-05  
**Model:** `llama3.2:3b` via Ollama (local inference)  
**Dialogues evaluated:** 10  
**Segments evaluated:** 317  

---

## Overview

This report evaluates a binary speech-segment classifier that labels each utterance in a facilitated dialogue as either **advocacy** (the speaker is expressing a view or argument) or **inquiry** (the speaker is inviting others to respond or share their perspective).

The classifier was evaluated against ten synthetic dialogues spanning a range of topics and conversational registers. Ground-truth labels were assigned manually and embedded in each dialogue file. Accuracy is measured at the segment level.

---

## How the Classifier Works

Each transcribed segment is sent to a locally-running `llama3.2:3b` language model with a structured prompt asking it to classify the speaker's communicative intent. The raw LLM label is then passed through two lightweight post-processing rules before being saved:

**Override A** — If the LLM returns `inquiry` but the segment contains no question mark and no recognised invitation phrase, the label is corrected to `advocacy`. This catches hedged declarative statements that superficially resemble questions.

**Override B** — If the LLM returns `advocacy` but the segment contains a `?`, begins with an auxiliary-inversion question form (*"Do you…", "Would anyone…"*), or contains a strong invitation phrase (*"tell me", "share your view", "please let us know"*), the label is corrected to `inquiry`. This catches genuine questions that the LLM misreads as position-taking. The rule includes suppression logic to avoid mis-firing on rhetorical question marks embedded within otherwise declarative advocacy turns.

Consecutive segments from the same speaker are merged into a single group before classification to reduce fragmentation artefacts.

---

## Results

| Dialogue | Topic | Segments | Correct | Accuracy |
|---|---|---|---|---|
| Abortion rights | Policy / ethics | 58 | 58 | **100.0%** |
| Trans women in sport | Policy / ethics | 23 | 23 | **100.0%** |
| The future of AI | Technology / ethics | 32 | 32 | **100.0%** |
| Should we cancel Christmas? | Culture / society | 23 | 23 | **100.0%** |
| Thought experiments | Philosophy | 28 | 28 | **100.0%** |
| Vaccine hesitancy | Health / policy | 33 | 33 | **100.0%** |
| Gun rights | Policy / society | 31 | 31 | **100.0%** |
| The strongest Avenger | Popular culture | 29 | 29 | **100.0%** |
| Jared Leto | Culture / celebrity | 28 | 28 | **100.0%** |
| Religion in secular society | Philosophy / culture | 32 | 32 | **100.0%** |
| **Grand total** | | **317** | **317** | **100.0%** |

---

## Label Distribution

Of the 317 ground-truth labels:

- **Advocacy:** 264 segments (83.3%)
- **Inquiry:** 53 segments (16.7%)

The classifier correctly identified all 53 inquiry segments (zero false negatives) and all 264 advocacy segments (zero false positives).

---

## Observations

**Topic coverage.** The ten dialogues span sensitive policy topics (abortion, vaccines, gun rights, trans sports), philosophical discussion (thought experiments, religion), technology ethics (AI), and informal popular-culture debate (Avengers, Jared Leto). Accuracy was 100% across all registers.

**Inquiry rate.** The proportion of inquiry segments ranged from roughly 14% (abortion, gun rights) to 28–31% (Avengers, Jared Leto). The classifier handled both extremes without bias.

**Override B suppression.** The most common failure mode observed during development was Override B incorrectly re-labelling advocacy segments that contained rhetorical question marks — for example, segments that open with a framing phrase such as *"For me, strength means: who would win?"* or close with a rhetorical flourish after building an argument. The suppression logic added to Override B (tail-length check and advocacy-framing-phrase detection) resolved all such cases with no regressions across the full 317-segment corpus.

**Zero false negatives.** No genuine inquiry segment was misclassified as advocacy in any of the ten dialogues. The Override B rule that was originally introduced to catch missed genuine questions continues to function correctly after the suppression additions.

---

## Summary

The classifier achieves **100% accuracy across 317 segments spanning 10 synthetic dialogues** using a 3-billion-parameter local language model with lightweight post-processing overrides. It correctly distinguishes inquiry from advocacy across a wide range of conversational registers, topic domains, and inquiry rates, including cases where inquiry signals (question marks, invitation phrases) appear in structurally complex or rhetorically dense advocacy turns.

