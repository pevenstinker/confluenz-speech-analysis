# Confluenz Speech Classifier — Technical Reference

**Project:** Confluenz Speech Analysis  
**Module:** `classifier.py`  
**Version:** April 2026  
**Language:** Python 3.10+

This document describes the complete classification pipeline: how it works, why each design decision was made, and how to configure and extend it. It is intended for researchers reproducing or extending this work, and for contributors to the open-source codebase.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Definitions](#2-definitions)
3. [Architecture](#3-architecture)
4. [Stage 1 — Segment Grouping](#4-stage-1--segment-grouping)
5. [Stage 2 — LLM Classification (Ollama backend)](#5-stage-2--llm-classification-ollama-backend)
6. [Stage 3 — Post-processing Overrides](#6-stage-3--post-processing-overrides)
7. [Regex Backend (fallback)](#7-regex-backend-fallback)
8. [Configuration](#8-configuration)
9. [Model Selection Guide](#9-model-selection-guide)
10. [What Is and Isn't Model-Specific](#10-what-is-and-isnt-model-specific)
11. [Accuracy Benchmarks](#11-accuracy-benchmarks)
12. [Known Limitations](#12-known-limitations)
13. [Extending the Classifier](#13-extending-the-classifier)

---

## 1. Overview

The Confluenz classifier assigns one of two labels to each segment of a transcribed group dialogue:

- **advocacy** — the speaker is stating an opinion, belief, or position
- **inquiry** — the speaker is asking a genuine question to invite others to respond

The pipeline has three stages that run in sequence for every group of segments:

```
Whisper segments → [1. Grouping] → [2. LLM] → [3. Post-processing overrides] → labelled segments
                                        ↕ (fallback if LLM unavailable)
                                  [Regex classifier]
```

Each stage addresses a specific failure mode identified during testing on real and synthetic Confluenz dialogues.

---

## 2. Definitions

These definitions come directly from the Confluenz facilitation framework and are embedded verbatim in the LLM prompt:

> **Advocacy:** The speaker states an opinion, belief, or position. They are declaring something, not inviting a response.

> **Inquiry:** The speaker asks a genuine question to invite others to respond. The intent is to hear, not to tell.

The key distinction is *communicative intent*, not grammatical form. A sentence can be syntactically interrogative but functionally advocacy ("Isn't it obvious that we need to act?"), and conversely a sentence with no question mark can be genuine inquiry ("Let me know what you all think"). The classifier is designed to target intent, not surface form — though surface form is used as a heuristic signal at all three stages.

**The default label is `advocacy`**, reflecting the empirical baseline that in most group discussions, the large majority of utterances are declarative. This is a soft prior, not an absolute rule.

---

## 3. Architecture

The pipeline is implemented in `classifier.py`. The entry point for external callers is:

```python
classify_segments(segments: list) -> tuple[str, str]
```

`segments` is a list of dicts produced by `transcriber.py`, each with at minimum a `"text"`, `"start"`, and `"end"` field. Labels are written in-place to `segment["label"]` and `segment["confidence"]`. The function returns `(backend, model)` — for example `("ollama", "llama3.2:3b")` or `("regex", "regex")` — which is recorded in the output JSON's `stats` block for reproducibility.

The backend is selected by the `CLASSIFIER` environment variable (`ollama` or `regex`). If `CLASSIFIER=ollama` but the Ollama service is not running, the pipeline falls back to the regex backend automatically and logs a warning.

---

## 4. Stage 1 — Segment Grouping

### The problem

The Whisper speech-to-text model segments audio by pausing, not by sentence. When a speaker does not pause between clauses, Whisper can split a single sentence across two or more segments:

```
Segment A: "I believe that I"            (ends at 13.26s)
Segment B: "believe that Israel is..."   (starts at 13.38s)
```

When the LLM sees segment B ("believe that") in isolation — as it would without grouping — it has no context to classify it correctly. In testing, isolated fragments like this were misclassified at a very high rate.

### The solution: `_group_segments()`

Before any LLM call, consecutive segments are merged into groups that represent complete thoughts. Two criteria trigger a merge:

1. **Lowercase start**: if a segment begins with a lowercase letter, it is treated as a continuation of the previous segment's sentence (Whisper capitalises the first word of new sentences).
2. **Short gap with no terminal punctuation**: if the gap between the end of segment N and the start of segment N+1 is less than 0.5 seconds, *and* segment N does not end with `.`, `!`, `?`, or `…`, the segments are merged.

```python
starts_lowercase = bool(text) and text[0].islower()
prev_has_terminal = bool(prev_text) and prev_text[-1] in ".!?…"
gap = seg["start"] - prev["end"]

if starts_lowercase or (not prev_has_terminal and gap < 0.5):
    groups[-1].append(seg)   # merge into current group
else:
    groups.append([seg])     # start new group
```

All segments in a group are assigned the same label after the group is classified as a whole using its combined text.

### Effect on accuracy

Prior to grouping, the 7-segment dialogue-45 test had 3 fragments that could not be correctly classified in isolation. After grouping, those fragments merged into one coherent advocacy sentence, which was correctly identified. The improvement was from 6/7 correct to 7/7 correct on that test.

### Configurable threshold

The 0.5-second gap threshold is a conservative heuristic derived from observation of Whisper's segmentation behaviour on conversational speech. It can be tightened (e.g. 0.3s) if over-merging is observed, or relaxed (e.g. 1.0s) for slower speech. It is currently hardcoded; a future version may expose it as a configuration parameter.

---

## 5. Stage 2 — LLM Classification (Ollama backend)

### What Ollama is

[Ollama](https://ollama.com) is a local LLM runtime. It runs large language models entirely on the user's own machine — no data is sent to any external service. This is essential for Confluenz's privacy requirements: recordings of sensitive group discussions never leave the local environment.

The Ollama Python client is used to send one prompt per group and parse the response.

### Per-group prompting

Each group's combined text is sent to the LLM as a separate chat message. This was a deliberate design choice after an earlier approach of batching all segments into a single JSON prompt failed: small models (≤1B parameters) defaulted to a single label for every item in the batch. Per-group prompting produces reliable results even with the smallest models.

### The prompt

Two prompt templates are used:

- **`_OLLAMA_PROMPT`** — used for the first group in a dialogue (no previous context available)
- **`_OLLAMA_PROMPT_WITH_CONTEXT`** — used for all subsequent groups; includes the text of the immediately preceding group labelled as "context only — do not classify this"

Both prompts:
- Define advocacy and inquiry precisely, using language from the Confluenz facilitation framework
- Provide **labelled examples** of both categories, including several counter-intuitive cases that were observed to cause misclassification during testing (see §6)
- State explicit numbered **key rules** covering the most common failure patterns
- Instruct the model to **reply with one word only** (`advocacy` or `inquiry`)

The one-word response format is critical for reliability with smaller models. Models with ≤3B parameters frequently deviate from structured output formats; asking for a single word yields a parseable response in nearly all cases. If the response does not contain either word, the segment falls back to the regex classifier.

### Confidence scoring

Confidence is assigned based on how cleanly the model responded:

| Response | Confidence |
|---|---|
| Exactly `"inquiry"` or `"inquiry."` | 0.90 |
| Contains "inquiry" with other words | 0.80 |
| Exactly `"advocacy"` or `"advocacy."` | 0.90 |
| Contains "advocacy" with other words | 0.80 |
| Neither word found | regex fallback |

These values are heuristic. They do not represent a calibrated probability and should not be interpreted as such. They are intended only to allow downstream filtering (e.g. the `--only-uncertain` flag in `label_file.py`).

---

## 6. Stage 3 — Post-processing Overrides

After the LLM assigns a label, two deterministic override functions are applied. These are the hardest-working part of the pipeline: they enforce linguistic constraints the LLM can reason about but sometimes fails to apply consistently.

### Override A: `_no_question_override()` — inquiry → advocacy

**Trigger:** LLM returned `inquiry`, but the combined text contains no `?` character and no recognised invitation phrase.

**Rationale:** During testing on real and synthetic dialogues, the most common failure pattern was the LLM labelling reflective, closing, or empathetic advocacy statements as inquiry. Examples:

- *"That resonates with me. I think this could be common ground."*
- *"I've found this valuable. This kind of engagement matters."*
- *"I hear that argument and I understand why it compels people, but..."*

These statements have conversational features — they engage with what someone else said, they are mild in tone — that superficially resemble inquiry. But they contain no question, and no invitation for others to speak. The override checks two conditions before flipping the label:

1. **No `?` in the text** — if there is any question mark, the inquiry label is preserved.
2. **No invitation phrase** — a curated set of phrases that constitute genuine invitations to respond even without a `?`: "let me know", "tell me", "I'd like to hear", "share your thoughts", "your opinion", "qu'en pensez-vous", etc.

If both conditions are met, the label is set to `advocacy` regardless of what the LLM said.

### Override B: `_force_inquiry_override()` — advocacy → inquiry

**Trigger:** LLM returned `advocacy`, but the combined text contains a clear syntactic inquiry signal.

**Rationale:** This is the complement to Override A. During testing on dialogue-48 (a real recording of naturally spoken speech), the LLM missed questions because:
- The question was embedded in a sentence with a declarative opening
- Whisper omitted the `?` punctuation from the transcript (a known Whisper limitation)
- The question began with auxiliary-inversion syntax ("Do you think...") which the LLM underweighted

Three conditions each independently trigger the flip to `inquiry`:

1. **Question mark present (excluding reported speech)** — if the text contains `?` and is not a reported speech construction, it is classified as inquiry. This is the strongest possible signal. Reported speech is detected by `_REPORTED_SPEECH_RE`, which matches patterns like "she goes like, ...", "he said, ...", "they were like, ...". Without this exclusion, a statement such as *"She goes like, are you going to call me or not?"* would be flipped to inquiry even though the speaker is narrating, not asking.
2. **Auxiliary-inversion at sentence start** — the pattern `^(but )?(do|does|did|would|could|should|can|will|is|are|...) (you|we|they|anyone|...)` detects inverted question syntax at the start of the combined text. This catches questions that Whisper transcribed without a `?`.
3. **Strong invitation phrase** — a set of phrases that unambiguously invite a response: "please let me know", "tell us know", "I'd like to hear", "if you have any thoughts", "please share your/what/which", "qu'en pensez-vous", etc. This is a stricter superset of the invitation phrases in Override A; it is designed to have very low false-positive rate.

### Override ordering

The overrides are applied in sequence:

```python
label = _no_question_override(combined, label)   # inquiry → advocacy
label = _force_inquiry_override(combined, label)  # advocacy → inquiry
```

Because Override A only acts when `label == "inquiry"` and Override B only acts when `label == "advocacy"`, they cannot conflict with each other. The final label is determined by whichever override fires last — but in practice each override only fires on segments where the LLM has already made a mistake.

---

## 7. Regex Backend (fallback)

The regex backend (`_classify_with_regex()`) is used when:
- `CLASSIFIER=regex` is set in `.env`
- Ollama is not running and the Ollama backend raises a `ConnectionError`
- An individual group returns an unrecognised response from the LLM (per-group fallback)

It uses two weighted pattern tables, `_INQUIRY` and `_ADVOCACY`, each containing ~25 bilingual (EN + FR) regex patterns. Each pattern carries a weight between 0.55 (weak signal) and 0.95 (very strong signal). The classification is based on the *peak* (highest-weight) matching pattern from each table, not the sum:

```python
inq = _peak_score(text, _INQUIRY)   # highest matching inquiry weight
adv = _peak_score(text, _ADVOCACY)  # highest matching advocacy weight
label = "inquiry" if inq >= adv else "advocacy"
confidence = inq / (inq + adv) if inq >= adv else adv / (inq + adv)
```

Peak scoring was chosen over cumulative scoring to prevent a single common word (e.g. "what") from overwhelming all other signals by appearing in many low-weight patterns simultaneously.

The regex backend also uses `_group_segments()` and applies the same post-processing overrides, so its accuracy benefits from the same structural improvements as the LLM backend.

**Known ceiling of the regex backend:** It achieves approximately 70–75% accuracy on real Confluenz dialogues. The most common failure pattern is that wh-words ("what", "where", "when") in declarative sentences trigger false-positive inquiry classifications. The LLM backend handles this correctly because it reasons about the *meaning* of the sentence rather than matching surface-level keywords.

---

## 8. Configuration

All configuration is set in `.env` in the project directory. The following variables affect the classifier:

| Variable | Default | Description |
|---|---|---|
| `CLASSIFIER` | `ollama` | Backend to use: `ollama` or `regex` |
| `OLLAMA_MODEL` | `gemma3:1b` | Ollama model name. Any model available via `ollama list` can be used. |
| `SAVE_TEXT_IN_JSON` | `true` | When `false`, transcribed text is used for classification but never written to any output file (privacy mode). Segments are still labelled; the `text` field is simply omitted from the saved JSON. |

The classifier backend and model name are recorded in the `stats` block of every output JSON file:

```json
"stats": {
  "classifierBackend": "ollama",
  "classifierModel": "llama3.2:3b",
  ...
}
```

This ensures every result file is self-documenting and reproducible.

---

## 9. Model Selection Guide

The `OLLAMA_MODEL` setting selects which locally-installed LLM is used. The following models have been tested:

### `gemma3:1b` (Google, ~600 MB)
- **Accuracy:** ~60% on real speech before prompt improvements; improved significantly with grouping and overrides
- **Speed:** Very fast on CPU (~2–3s per segment)
- **Recommendation:** Not recommended for research use. Even with all improvements applied, this model is too small to reliably handle nuanced advocacy. It defaults to a single label under batch prompting (a documented failure mode of 1B models). Suitable only for offline testing or extremely resource-constrained environments.
- **Notes:** Requires the simple one-word prompt format. Multi-item or structured prompts cause consistent label collapse.

### `llama3.2:3b` (Meta, ~2 GB)
- **Accuracy:** 82.8% baseline → 100% on 58-segment synthetic test after all improvements; strong performance on short real recordings
- **Speed:** ~3–5s per segment on CPU
- **Recommendation:** Recommended default for research use. Handles nuanced advocacy and reflective speech well. Responds reliably to the one-word format.
- **Pull command:** `ollama pull llama3.2:3b`

### `llama3.1:8b` (Meta, ~4.7 GB)
- **Accuracy:** Not yet benchmarked on Confluenz data
- **Speed:** ~8–15s per segment on CPU
- **Recommendation:** Expected to outperform the 3B model on ambiguous cases. Suggested if 3B accuracy is insufficient for research requirements after collecting labelled data from real sessions.
- **Pull command:** `ollama pull llama3.1:8b`

To switch models, change `OLLAMA_MODEL` in `.env`. No code changes are required.

---

## 10. What Is and Isn't Model-Specific

This is an important consideration for reproducibility. The pipeline has components at different levels of model-specificity:

### Model-agnostic (work with any LLM)

- **Segment grouping** (`_group_segments`): purely structural, operates on Whisper timestamps and capitalisation. Independent of the LLM entirely.
- **Override A — `_no_question_override`**: a deterministic regex check. Applied after the LLM but has no dependence on which LLM was used.
- **Override B — `_force_inquiry_override`**: same — deterministic regex, LLM-independent.
- **Regex backend**: no LLM dependency whatsoever.
- **The one-word response format**: works with any instruction-following LLM. The parser searches for "inquiry" or "advocacy" anywhere in the response, so models that add punctuation or a trailing sentence still work correctly.

### Prompt-tuned for instruction-following models in general

- **The prompt examples and key rules**: these were written for the class of instruction-following models (models trained to follow task descriptions). They are not specific to Llama or any particular model family. They should port directly to GPT-4, Claude, Mistral, Qwen, etc.
- **The context-aware prompt** (`_OLLAMA_PROMPT_WITH_CONTEXT`): the "previous segment as context" pattern is a general prompting technique. It is not Llama-specific.

### Effectively calibrated against `llama3.2:3b`

- **The specific advocacy and inquiry examples in the prompt**: these were iteratively developed by identifying cases where `llama3.2:3b` made errors and adding counter-examples. While the examples encode generally valid linguistic reasoning, the *selection* of which examples to include was driven by `llama3.2:3b`'s specific failure modes.
  - A different model family might have different systematic errors (e.g. Mistral models tend to over-classify as inquiry; models fine-tuned on customer-service data may have a different baseline).
  - When switching to a substantially different model (e.g. a non-Meta model, or a model fine-tuned on a different domain), it is advisable to re-run the synthetic test dialogue and check for new systematic errors.

### Summary table

| Component | Model-specific? | Notes |
|---|---|---|
| `_group_segments()` | No | Whisper output structure only |
| `_no_question_override()` | No | Deterministic regex |
| `_force_inquiry_override()` | No | Deterministic regex |
| Regex backend | No | No LLM involved |
| One-word response format | No | Works with any instruction model |
| Prompt examples & rules | Calibrated against `llama3.2:3b` | May need re-tuning for other models |
| Context-aware prompt structure | No | General prompting technique |
| `OLLAMA_MODEL` config | Yes | Model-specific setting |

---

## 11. Accuracy Benchmarks

### Synthetic test dialogue (dialogue ID 98)

A synthetic 15-minute dialogue with 10 speakers discussing abortion was created with hand-labelled ground-truth. It was designed to include ~10% inquiry and to exercise the specific failure modes identified during development (reflective advocacy, declarative "What..." sentences, empathetic rebuttals, closing statements).

| Configuration | Correct / 58 | Accuracy |
|---|---|---|
| `llama3.2:3b`, no improvements (baseline) | 48 / 58 | 82.8% |
| + prompt counter-examples (5a) + no-question override (5b) + previous-segment context (5d) | 58 / 58 | **100%** |
| + force-inquiry override (5c) | 58 / 58 | **100%** (no regression) |

### Round 2 synthetic dialogues

Four additional synthetic dialogues were created covering diverse conversational topics, each with hand-labelled ground truth. Results using `llama3.2:3b` after all improvements:

| Dialogue | Topic | Segments | Correct | Accuracy |
|---|---|---|---|---|
| Synthetic AI | Future of AI | 32 | 31 | 96.9% |
| Synthetic Christmas | "Cancel Christmas" | 23 | 22 | 95.7% |
| Synthetic trans sports | Trans women in sport | 23 | 23 | **100.0%** |
| Synthetic thought experiments | Philosophical thought experiments | 28 | 27 | 96.4% |
| **Round 2 total** | | **106** | **103** | **97.2%** |

**Cumulative across all 5 synthetic dialogues (164 segments): 161/164 = 98.2%.**

The 3 remaining errors all share the same root cause: Override B (`_force_inquiry_override`) fires on a `?` inside a rhetorical or illustrative embedded question — cases where the LLM correctly returned `advocacy` but the `?` rule overruled it. These were analysed and deferred: fixing them would require distinguishing rhetorical from genuine questions at the syntactic level, which carries a high regression risk. See `classification-report-round2.md` for the full error analysis.

One error (thought experiments, Round 2 initial run) was fixed: the facilitator closing phrase "share **which** thought experiment" was not matched by `_STRONG_INVITATION_RE`. Adding `which` to the `share` alternation corrected this without any regression.

### Real recording — dialogue-45

A 35-second real recording used as the primary real-speech test case during development.

| Configuration | Correct / 7 | Notes |
|---|---|---|
| `gemma3:1b`, batch prompt | 0 / 7 | All labelled advocacy; batch prompt failure |
| `gemma3:1b`, per-segment prompt | 6 / 7 | Fragment segments misclassified in isolation |
| `llama3.2:3b`, with grouping + all improvements | 7 / 7 | All correct |

### Important caveat

The synthetic test dialogue was produced by the same authors who designed the classifier and prompt. It therefore reflects known failure modes rather than providing an independent test of generalisation. Accuracy on novel real-world Confluenz sessions may differ. Collection of labelled data from real sessions via `label_file.py` is recommended before drawing quantitative conclusions for research publication.

---

## 12. Known Limitations

### Whisper punctuation omission

Whisper frequently omits question marks from transcriptions of conversational speech, particularly when the speaker's intonation does not strongly mark the question. Override B (`_force_inquiry_override`) mitigates this for questions that begin with auxiliary-inversion syntax ("Do you think..."), but questions with declarative syntax and rising intonation only ("You think that's right?") will lose their question mark and may not be caught by the override.

### Fragment groups may share an incorrect label

When multiple Whisper fragments are merged into one group, all segments in the group receive the same label. If the group spans both an advocacy statement and a question (e.g. "I believe this is wrong. What do you think?"), the `?` in the text will cause Override B to label the whole group as inquiry, even though part of it is advocacy. This is a known trade-off: grouping dramatically improves accuracy on pure fragments but introduces a coarsening effect on mixed-content groups.

### Bilingual (EN + FR) support is asymmetric

The regex patterns and LLM prompt examples provide fuller coverage of English than French. The LLM's training data is also predominantly English. French Confluenz sessions should be expected to underperform English sessions until French-specific prompt examples are added and validated on French test data. The invitation-phrase regexes include a small number of French patterns (`est-ce que`, `qu'en pensez-vous`, `à votre avis`, etc.) but coverage is incomplete.

### Post-processing overrides can conflict with nuanced speech

The overrides are deterministic and do not reason about meaning. A sentence like "Would that be fair?" is genuinely an inquiry, but if Whisper transcribes it as "Would that be fair." (no `?`), Override B will not fire and the LLM must classify it correctly on its own. Conversely, a rhetorical question like "Would anyone seriously argue that?" may be classified as inquiry when it is functionally advocacy. These are edge cases but they arise in real dialogues.

### Reported speech (partial mitigation)

When a speaker narrates a question someone else asked — e.g. *"She goes like, are you going to call me or not?"* — the `?` belongs to the third-party quote, not the current speaker. `_REPORTED_SPEECH_RE` detects the most common English reported-speech verbs ("goes like", "said", "asked", "was/were like") and suppresses the `?`-flip in those cases. Less common constructions ("turns to me and asks, ...") are not currently covered and may still be misclassified as inquiry.

---

## 13. Extending the Classifier

### Adding new invitation phrases

Edit `_INVITATION_RE` and/or `_STRONG_INVITATION_RE` in `classifier.py`. `_INVITATION_RE` is used by Override A to preserve an existing inquiry label; `_STRONG_INVITATION_RE` is used by Override B to force inquiry from advocacy. Add phrases that are genuinely unambiguous invitations to respond, not just polite-sounding language. Note that the `share` branch currently matches `share your`, `share what`, and `share which` — if new "share X" constructions emerge, extend the alternation rather than adding a separate pattern.

### Adding new prompt examples

Edit `_OLLAMA_PROMPT` and `_OLLAMA_PROMPT_WITH_CONTEXT` in `classifier.py`. Both prompts must be kept in sync. When adding an example, note whether it is correcting a specific model's failure or encoding a generally valid linguistic rule — this distinction matters for reproducibility documentation.

### Using a different LLM provider (e.g. OpenAI, Anthropic)

Replace `_classify_with_ollama()` with a function that calls your preferred API. The prompt format, grouping, and override logic are all provider-agnostic. The function signature must return `"ollama"` (or a new backend string) on success and `"regex"` on failure, matching the return contract of `classify_segments()`.

### Building a fine-tuned classifier

Use `label_file.py` to collect human-corrected labels from real Confluenz sessions into `training-data.jsonl`. Once ~300–500 labelled examples are available, a scikit-learn `TfidfVectorizer + LogisticRegression` pipeline trained on this data will likely outperform the general-purpose LLM on in-domain speech. See the `train.py` placeholder in the project for the intended training interface.

### Adjusting the grouping threshold

Change the `0.5` constant in `_group_segments()` to modify the maximum gap (in seconds) that triggers a fragment merge. Shorter values reduce over-merging at the cost of more fragments reaching the LLM in isolation.
