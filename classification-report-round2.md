# Classifier Evaluation — Round 2 (Four New Synthetic Dialogues)
**Date:** 2026-04-04  
**Model:** `llama3.2:3b` via Ollama  
**Dialogues tested:** AI futures, Cancel Christmas, Trans women in sport, Thought experiments  
**Total segments:** 106  

---

## Summary Results

| Dialogue | Segments | Correct | Accuracy |
|---|---|---|---|
| The future of AI | 32 | 31 | 96.9% |
| Should we cancel Christmas? | 23 | 22 | 95.7% |
| Trans women competing in sport | 23 | 23 | **100.0%** |
| Thought experiments | 28 | 26 | 92.9% |
| **Grand total** | **106** | **102** | **96.2%** |

Combined with the synthetic abortion dialogue (58/58, 100%), the classifier now scores **160/164 correct (97.6%)** across 5 diverse synthetic test dialogues.

---

## The 4 Misclassifications

### Error 1 — AI futures, Leon  
**Direction:** advocacy → inquiry (false positive)  
**Text:**
> "But whose values? That question doesn't go away just by adding the word alignment. The values baked into the most powerful AI systems will reflect the priorities of whoever builds them. Right now that's a very narrow slice of humanity. I believe that's a problem that needs to be solved before we worry about aligning the AI itself."

**What happened:** The text contains `?` ("But whose values?"). Override B (`_force_inquiry_override`) unconditionally flips any `"advocacy"` label to `"inquiry"` whenever a `?` is present. The LLM had correctly identified this as advocacy; the override overruled it.

**Root cause:** The `?` is a **rhetorical question** — a debating device used to introduce a point, not to solicit a response. The override cannot distinguish rhetorical from genuine questions at the surface level.

---

### Error 2 — Cancel Christmas, Younes  
**Direction:** advocacy → inquiry (false positive)  
**Text:**
> "That's a real phenomenon — seasonal isolation and what clinicians call the holiday effect on mental health. But I'd ask: is Christmas the cause, or is it a lens that makes existing inequalities in connection and belonging more visible? Removing Christmas wouldn't remove loneliness."

**What happened:** Same as Error 1. The embedded `?` ("is Christmas the cause...?") triggered Override B. The LLM correctly said advocacy; the override flipped it. Again a **rhetorical embedded question** mid-paragraph, not a genuine invitation for others to respond.

**Aggravating factor:** The segment ends with "Removing Christmas wouldn't remove loneliness." — a strong declarative advocacy close — but by the time the code reaches that point the label has already been set.

---

### Error 3 — Thought experiments, Nadia  
**Direction:** advocacy → inquiry (false positive)  
**Text:**
> "My view is that it matters enormously, but not for metaphysical reasons — for practical and legal ones. Who owns the ship of Theseus? What counts as the original artefact for purposes of cultural heritage law? When does a company remain legally responsible for a predecessor's liabilities? These questions have real answers that depend on how we draw the identity line."

**What happened:** Same mechanism. Multiple `?` marks appear in the middle of the text, each a rhetorical illustration of a position being argued. The segment *opens* with "My view is..." and *closes* with "These questions have real answers..." — both strong advocacy markers. But Override B fires on the first `?` it encounters.

**Note:** This is the most instructive case because the speaker is explicitly advocating a position *by invoking* questions as examples. "These questions have real answers" is the tell — a genuine inquiry speaker would not say that.

---

### Error 4 — Thought experiments, Facilitator  
**Direction:** inquiry → advocacy (false negative)  
**Text:**
> "To close, I'd ask everyone to share which thought experiment they find most genuinely illuminating — the one that most changed how you think. You don't have to justify it. Just share which one stays with you."

**What happened:** The LLM labelled this advocacy. Override A (`_no_question_override`) correctly left it alone because there is no `?`. Override B did not fire because the text contains no `?` and none of the `_STRONG_INVITATION_RE` patterns matched.

**Root cause:** The invitation phrase is **"I'd ask everyone to share"**. The current `_STRONG_INVITATION_RE` covers `share\s+(your|what)` but not `share\s+which`. And `i'd ask` is not in any pattern at all.

---

## Root Cause Analysis

Errors 1–3 share one root cause. Error 4 is a separate, unrelated issue.

### Errors 1–3: Override B is too broad on `?`

The `?` branch of `_force_inquiry_override` was introduced to catch genuine questions that the LLM missed — specifically "Do you think...?" type questions that Whisper sometimes transcribes without a question mark, which led to misclassifications in dialogue-48. That fix was correct and important.

The unintended consequence is that the rule cannot distinguish between:

| Type | Example | Intent |
|---|---|---|
| Genuine question | "Do you think the answer is to slow AI development?" | Soliciting a response |
| Rhetorical embedded question | "But whose values? That question doesn't disappear." | Introducing an argument |
| Illustrative series | "Who owns the ship? What counts as the original?" | Building a case |

All three types contain `?`. The genuine question is the only one that should be labelled inquiry. Surface-level pattern matching on `?` alone cannot reliably distinguish them.

### Error 4: Missing invitation pattern

The `_STRONG_INVITATION_RE` pattern `share\s+(your|what)` does not cover `share\s+which`, and the phrase "I'd ask everyone to share" is not represented at all. This is a coverage gap rather than a design flaw.

---

## Improvement Analysis

### Error 4 fix — safe, no regression risk

Adding `share\s+which` to the `share` branch of `_STRONG_INVITATION_RE` (changing `(your|what)` to `(your|what|which)`) would directly fix this case. The phrase "share which X" in natural speech almost always signals an invitation. It is difficult to construct a genuine advocacy statement that contains this phrase.

Alternatively, adding a pattern for `\bi[''']d ask\s+(everyone|you\s+all|all\s+of\s+you)\b` would be equally targeted and safe.

**Recommendation: implement this fix.** Zero meaningful regression risk.

### Errors 1–3 fix — risky, should be deferred

Any fix robust enough to suppress Override B's `?` rule on rhetorical questions would require reasoning about whether the `?` is the *communicative purpose* of the utterance or merely incidental to it — which is exactly the kind of reasoning the LLM does (and got right in all three cases before the override intervened).

Candidate approaches and their risks:

| Approach | Risk |
|---|---|
| Remove the `?` branch from Override B entirely | **High** — reverts dialogue-48 regressions; "Do you think...?" without `?` would be missed |
| Only trigger if `?` is within the last N characters | **Medium** — threshold is arbitrary; Younes (Error 2) ends 45 chars after `?`, hard to exclude cleanly |
| Only trigger if text is short (< 120 chars) | **Medium** — loses protection for long genuine questions |
| Add patterns for "My view is", "I believe", "I argue" as suppression signals | **Medium** — would suppress genuine inquiry that follows an advocacy opening sentence |

None of these is clean. All trade one kind of error for another.

**Recommendation: do not fix Errors 1–3 now.** The classifier correctly identified 3 out of these same 4 cases at the LLM level. The override is what introduced the error. The most reliable fix would be a prompt-level adjustment — training the LLM to respond `"advocacy"` when the question is clearly rhetorical — but this risks breaking the cases the override was added to protect. Leave as-is until real-world labelled data justifies a targeted change.

---

## What "Good Place" Means in Numbers

The absolute error rate across the 5 synthetic dialogues is **4/164 = 2.4%**. All 4 errors are in philosophically interesting edge cases — rhetorical questions and soft closing invitations — that are genuinely difficult even for human raters.

The classifier achieved 100% on both the original abortion dialogue and the trans women in sport dialogue, two of the more sensitive and linguistically nuanced topics.

The 4 remaining errors are well understood. Three of them are caused by a known, deliberate design decision (Override B) that fixed more errors than it introduced. One is a trivial coverage gap in a pattern list.

---

## Proposed Next Action

Apply only the Error 4 fix:

```python
# In _STRONG_INVITATION_RE, change:
r"|\b(please\s+)?share\s+(your|what)\b"
# To:
r"|\b(please\s+)?share\s+(your|what|which)\b"
```

Then re-run the eval script to confirm 103/106 (97.2%) with no regressions.

Defer Errors 1–3 until real-world labelled data is available, at which point a prompt-level adjustment (additional counter-examples) would be a safer intervention than changing the override logic.
