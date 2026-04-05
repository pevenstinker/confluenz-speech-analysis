"""
Bilingual (EN + FR) advocacy-vs-inquiry classifier.

Definitions (from Confluenz):
  inquiry  — asking a genuine question to the group; open, curious, inviting response
  advocacy — stating an opinion, belief, or position in a declarative way

Two backends (controlled by config.classifier):
  'ollama' — sends all segments in one LLM call (default, more accurate)
             Falls back to regex automatically if Ollama is not running.
  'regex'  — fast rule-based fallback, no external dependencies
"""

import json
import re
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Pattern tables  (pattern_string, weight)
# ---------------------------------------------------------------------------

# Inquiry signals — questions, invitations to respond, curiosity markers
_INQUIRY: List[Tuple[str, float]] = [
    # Question mark is the strongest single signal
    (r"\?",                                                                  0.90),

    # Direct "what do you think / feel / believe" invitations
    (r"what do you (think|feel|believe|reckon|say|make of)",                 0.95),
    (r"what('s| is| are) your (thought|view|opinion|take|perspective|stance)", 0.95),
    (r"qu['']en (penses?-tu|pensez-vous|dis-tu|dites-vous)",                 0.95),
    (r"(c'est |c'était )?quoi votre (avis|opinion|point de vue)",            0.90),

    # EN — wh-words at sentence start
    (r"(?:^|\.\s*|\?\s*)(what|how|why|when|where|who|which|whose|whom)\b",  0.80),
    # EN — wh-words mid-sentence (lower weight — could be embedded clause)
    (r"\b(what|how|why|when|where|who|which)\b",                            0.55),

    # EN — auxiliary inversion questions
    (r"\bdo (you|we|they|anyone|people)\b",                                  0.70),
    (r"\bdoes (anyone|everybody|someone)\b",                                 0.70),
    (r"\bdid (you|we|anyone)\b",                                             0.65),
    (r"\bdon[''']t you\b|\bwouldn[''']t you\b|\bshouldn[''']t\b",           0.70),
    (r"\b(is|are|was|were|will|would|should|could|can|have|has) (you|anyone|we|they|it|that|this)\b", 0.55),

    # EN — open invitations
    (r"\banyone (have|know|think|feel|want|willing)\b",                      0.80),
    (r"\bwhat about\b|\bhow about\b",                                        0.65),
    (r"\bwhat if\b|\bhow come\b",                                            0.65),

    # FR — question markers
    (r"\best[- ]ce que\b",                                                   0.90),
    (r"\bqu['']est[- ]ce (que?|qui)\b",                                      0.90),
    (r"(?:^|\.\s*)(comment|pourquoi|quand|où|qui|quoi|lequel|laquelle|lesquels|lesquelles|quel|quelle|quels|quelles)\b", 0.80),
    (r"\b(comment|pourquoi|quand|qui)\b",                                    0.55),

    # FR — auxiliary inversion / verb-subject inversion
    (r"\bpensez[- ]vous\b|\bcroyez[- ]vous\b|\btrouvez[- ]vous\b",          0.85),
    (r"\bà votre avis\b|\bselon vous\b|\bà ton avis\b|\bselon toi\b",       0.90),
    (r"\bavez[- ]vous\b|\bêtes[- ]vous\b|\bfaites[- ]vous\b",               0.75),

    # FR — open invitations
    (r"\bquelqu['']un (sait|pense|croit|a|peut)\b",                         0.80),
    (r"\bqu['']en pensez\b|\bqu['']en penses\b",                            0.90),
]

# Advocacy signals — opinions, positions, declarations
_ADVOCACY: List[Tuple[str, float]] = [
    # EN — "I think / believe / feel / know"
    (r"\bI (think|believe|feel|know|find|consider|maintain|argue|hold)\b",   0.75),
    (r"\bI('m| am) (sure|certain|convinced|confident|positive)\b",           0.80),
    (r"\bin my (opinion|view|experience|judgment|humble opinion)\b",         0.90),
    (r"\bfrom my (perspective|point of view|standpoint|experience)\b",       0.88),
    (r"\bto me\b|\bif you ask me\b|\bpersonally\b",                         0.70),

    # EN — strong declarative stances
    (r"\bwe (should|must|need to|have to|ought to)\b",                      0.75),
    (r"\bone (should|must|has to|ought to)\b",                              0.72),
    (r"\bit[''']s (clear|obvious|evident|true|false|certain|undeniable|a fact)\b", 0.78),
    (r"\bthe fact (is|remains)\b|\bthe reality (is|remains)\b|\bthe truth is\b", 0.82),
    (r"\bobviously\b|\bclearly\b|\bundoubtedly\b|\bwithout (a )?doubt\b",   0.70),

    # EN — preference / sentiment declarations
    (r"\bI (love|hate|like|dislike|prefer|support|oppose|enjoy|detest|appreciate|admire)\b", 0.65),
    (r"\bthat[''']s (right|wrong|correct|incorrect|true|false|great|terrible|good|bad)\b", 0.70),
    (r"\bI (agree|disagree|concur|object)\b",                               0.72),

    # FR — "je pense / crois / trouve"
    (r"\bje (pense|crois|trouve|estime|considère|soutiens|maintiens|argue)\b", 0.75),
    (r"\bje suis (sûr|certain|convaincu|persuadé|confiant)\b",              0.80),
    (r"\bà mon avis\b|\bselon moi\b|\bde mon (point de vue|côté|expérience)\b|\bpour moi\b", 0.90),

    # FR — strong declarative stances
    (r"\b(il faut|on doit|on devrait|nous devons|nous devrions|il faudrait)\b", 0.75),
    (r"\bc[''']est (clair|évident|vrai|faux|certain|indéniable|un fait)\b", 0.78),
    (r"\bévidemment\b|\bclairement\b|\bsans (aucun )?doute\b",              0.70),

    # FR — preference / sentiment declarations
    (r"\bj['']aime\b|\bje n['']aime pas\b|\bje préfère\b|\bje déteste\b",   0.65),
    (r"\bj['']apprécie\b|\bje soutiens\b|\bje m['']oppose\b",               0.65),
    (r"\bje suis (d['']accord|en désaccord)\b",                             0.72),
]

_DEFAULT_LABEL = "advocacy"
_DEFAULT_CONFIDENCE = 0.55  # slight default lean toward advocacy (statements are more common)


def _peak_score(text: str, patterns: List[Tuple[str, float]]) -> float:
    """Return the highest weight among all matching patterns (not cumulative)."""
    t = text.lower()
    best = 0.0
    for pat, weight in patterns:
        if re.search(pat, t, re.IGNORECASE):
            if weight > best:
                best = weight
    return best


def classify(text: str) -> Tuple[str, float]:
    """Classify *text* as 'advocacy' or 'inquiry' using regex rules."""
    if not text or not text.strip():
        return _DEFAULT_LABEL, _DEFAULT_CONFIDENCE

    inq = _peak_score(text, _INQUIRY)
    adv = _peak_score(text, _ADVOCACY)

    if inq == 0.0 and adv == 0.0:
        return _DEFAULT_LABEL, _DEFAULT_CONFIDENCE

    total = inq + adv
    if inq >= adv:
        return "inquiry", round(inq / total, 3)
    else:
        return "advocacy", round(adv / total, 3)


def _group_segments(segments: list) -> list:
    """Group consecutive Whisper segments that form a single sentence/thought.

    Whisper emits lowercase-starting fragments when a sentence spans two segments.
    Combining them into one classification unit gives the model (and regex) the
    full context of the thought.  A very short gap (<0.5 s) with no terminal
    punctuation is also treated as a continuation.

    All segments in a group receive the same label after classification.
    """
    if not segments:
        return []

    groups: list = [[segments[0]]]

    for prev, seg in zip(segments, segments[1:]):
        text = (seg.get("text") or "").strip()
        prev_text = (prev.get("text") or "").strip()
        gap = seg.get("start", 0) - prev.get("end", 0)
        prev_has_terminal = bool(prev_text) and prev_text[-1] in ".!?…"
        starts_lowercase = bool(text) and text[0].islower()

        if starts_lowercase or (not prev_has_terminal and gap < 0.5):
            groups[-1].append(seg)
        else:
            groups.append([seg])

    return groups


def _classify_with_regex(segments: list) -> None:
    """Classify all segments in-place using the regex backend, grouping fragments."""
    for group in _group_segments(segments):
        combined = " ".join((s.get("text") or "").strip() for s in group).strip()
        label, confidence = classify(combined)
        for seg in group:
            seg["label"] = label
            seg["confidence"] = confidence


def classify_segments(segments: list, prev_segment: dict = None) -> tuple:
    """Classify each segment dict in-place.

    Parameters
    ----------
    segments:     list of segment dicts to classify (modified in-place)
    prev_segment: optional last segment from the previous chunk, used as
                  context for the first group in this call (cross-chunk continuity)

    Returns (backend, model) where backend is 'ollama' or 'regex' and model
    is the Ollama model name (e.g. 'gemma3:1b') or 'regex' for the rule-based backend.
    Segments without a 'text' field (privacy mode) are skipped."""
    from config import config

    active_segments = [s for s in segments if s.get("text")]

    # Mark privacy-mode segments (no text) with neutral defaults
    for seg in segments:
        if not seg.get("text"):
            seg["label"] = seg.get("label") or "advocacy"
            seg["confidence"] = seg.get("confidence") or 0.0

    if not active_segments:
        return "regex", "regex"

    if config.classifier == "ollama":
        backend = _classify_with_ollama(active_segments, prev_segment=prev_segment)
    else:
        backend = "regex"

    if backend == "regex":
        _classify_with_regex(active_segments)

    model = config.ollama_model if backend == "ollama" else "regex"
    return backend, model


# ---------------------------------------------------------------------------
# Ollama backend — groups fragments into sentences, then classifies each group
# ---------------------------------------------------------------------------

_OLLAMA_PROMPT = """\
You are helping classify speech in a group discussion.

ADVOCACY: the speaker states an opinion, belief, or position. They are declaring something, not asking.
INQUIRY: the speaker asks a genuine question to invite others to respond.

ADVOCACY examples:
- "I think this is wrong"
- "I'm making a statement. I believe that people deserve peace"
- "I believe that this is a genocide and people deserve the right to safety"
- "We should do something about this"
- "Personally, I feel that this situation is unjust"
- "Je crois que nous devons agir"
- "What this conversation ignores is the lived reality of women" (declarative — starts with What but is NOT a question)
- "What I want to push back on is the assumption that..." (declarative — starts with What but is NOT a question)
- "That resonates with me. I think this could be common ground." (reflective agreement — advocacy)
- "I've found this valuable. This kind of engagement matters." (closing statement — advocacy)
- "I want to close by saying that I hope people on both sides..." (declarative closing — advocacy)
- "I hear that argument and I understand why it compels people, but..." (empathetic rebuttal — advocacy)

INQUIRY examples:
- "What do you think about this?"
- "Do you agree with that?"
- "How does this affect you? Let me know"
- "Are the actions justified, or are they unjust? What does everyone think?"
- "Qu'en pensez-vous?"
- "For those of you who are pro-life — how do you think about cases of rape?"

Key rules:
1. If the speaker uses "I believe", "I think", "I feel", or says what people "deserve" or "should" have — that is ADVOCACY.
2. A sentence beginning with "What" is ADVOCACY unless it is a genuine question: "What do you think?" is inquiry; "What this shows is..." is advocacy.
3. Reflective, closing, or empathetic statements without a question mark are ADVOCACY.
4. Only classify as INQUIRY if the speaker is genuinely inviting others to respond with their own views.

Classify the following. Reply with ONLY one word: advocacy or inquiry

Text: "{text}\""""


_OLLAMA_PROMPT_WITH_CONTEXT = """\
You are helping classify speech in a group discussion.

ADVOCACY: the speaker states an opinion, belief, or position. They are declaring something, not asking.
INQUIRY: the speaker asks a genuine question to invite others to respond.

ADVOCACY examples:
- "I think this is wrong"
- "I'm making a statement. I believe that people deserve peace"
- "I believe that this is a genocide and people deserve the right to safety"
- "We should do something about this"
- "Personally, I feel that this situation is unjust"
- "Je crois que nous devons agir"
- "What this conversation ignores is the lived reality of women" (declarative — starts with What but is NOT a question)
- "What I want to push back on is the assumption that..." (declarative — starts with What but is NOT a question)
- "That resonates with me. I think this could be common ground." (reflective agreement — advocacy)
- "I've found this valuable. This kind of engagement matters." (closing statement — advocacy)
- "I want to close by saying that I hope people on both sides..." (declarative closing — advocacy)
- "I hear that argument and I understand why it compels people, but..." (empathetic rebuttal — advocacy)

INQUIRY examples:
- "What do you think about this?"
- "Do you agree with that?"
- "How does this affect you? Let me know"
- "Are the actions justified, or are they unjust? What does everyone think?"
- "Qu'en pensez-vous?"
- "For those of you who are pro-life — how do you think about cases of rape?"

Key rules:
1. If the speaker uses "I believe", "I think", "I feel", or says what people "deserve" or "should" have — that is ADVOCACY.
2. A sentence beginning with "What" is ADVOCACY unless it is a genuine question: "What do you think?" is inquiry; "What this shows is..." is advocacy.
3. Reflective, closing, or empathetic statements without a question mark are ADVOCACY.
4. A segment that responds to or builds on what was just said is usually ADVOCACY, not inquiry.
5. Only classify as INQUIRY if the speaker is genuinely inviting others to respond with their own views.

Previous segment (context only — do not classify this):
"{prev_text}"

Classify the segment below. Reply with ONLY one word: advocacy or inquiry

Text: "{text}\""""


# Phrases that signal a genuine invitation to respond (used by 5b override).
_INVITATION_RE = re.compile(
    r"\blet me know\b|\btell me\b|\bi[''']d like to (hear|know)\b"
    r"|\bshare your\b|\byour (thoughts|views?|opinion|perspective)\b"
    r"|\bqu['']en (penses?-tu|pensez-vous)\b",
    re.IGNORECASE,
)

# Strong invitation phrases — used by 5b to preserve inquiry AND by 5c to force inquiry.
_STRONG_INVITATION_RE = re.compile(
    r"\b(please\s+)?(let|tell)\s+(me|us)\s+know\b"
    r"|\bi[''']d like to (hear|know)\b"
    r"|\bi want to (hear|know)\b"
    r"|\b(please\s+)?share\s+(your|what|which)\b"
    r"|\bif you have (any\s+)?(thoughts|questions|feedback|comments)\b"
    r"|\bqu['']en (penses?-tu|pensez-vous)\b",
    re.IGNORECASE,
)

# Reported speech patterns: "she goes like, X?", "he said, X?", "they were like, X?"
# Used to suppress the ?-based override when the question mark belongs to a quoted third party.
_REPORTED_SPEECH_RE = re.compile(
    r"\b(he|she|they|we|you)\s+go(es)?(\s+like)?\s*,"
    r"|\b(he|she|they|i|we|you)\s+said\s*,"
    r"|\b(he|she|they|i|we|you)\s+asked\s*,"
    r"|\b(he|she|they|i|we|you)\s+(was|were)\s+like\s*,",
    re.IGNORECASE,
)

# Auxiliary-inversion question starts: "Do you", "Would anyone", "Can we", etc.
_QUESTION_START_RE = re.compile(
    r"^(but\s+)?(do|does|did|would|could|should|have|has|haven[''']t"
    r"|doesn[''']t|don[''']t|didn[''']t|wouldn[''']t|couldn[''']t"
    r"|shouldn[''']t|can|will|won[''']t|is|are|was|were)"
    r"\s+(you|we|they|anyone|everyone|somebody|people|it|that|this)\b",
    re.IGNORECASE,
)


def _no_question_override(text: str, label: str) -> str:
    """5b: If the LLM returned 'inquiry' but the text contains no question mark and
    no recognised invitation phrase, flip the label to 'advocacy'.
    This catches reflective/closing statements that superficially resemble questions."""
    if label != "inquiry":
        return label
    if "?" in text:
        return label
    if _INVITATION_RE.search(text):
        return label
    return "advocacy"


def _force_inquiry_override(text: str, label: str) -> str:
    """5c (complement to 5b): If the LLM returned 'advocacy' but the text contains a
    clear inquiry signal, flip the label to 'inquiry'.
    Handles: question marks the LLM ignored, auxiliary-inversion question starts
    (\"Do you think...\"), and strong invitation phrases (\"please let me know\")."""
    if label != "advocacy":
        return label
    if "?" in text and not _REPORTED_SPEECH_RE.search(text):
        return "inquiry"
    if _STRONG_INVITATION_RE.search(text):
        return "inquiry"
    if _QUESTION_START_RE.match(text.strip()):
        return "inquiry"
    return label


def _classify_with_ollama(segments: list, prev_segment: dict = None) -> str:
    """Classify segments using Ollama, grouping fragments before sending to the model.

    Parameters
    ----------
    segments:     list of segment dicts to classify in-place
    prev_segment: optional last segment from the previous chunk, used as
                  initial context for the first group (cross-chunk continuity)

    Returns 'ollama' on success, 'regex' on failure."""
    from config import config
    try:
        import ollama
    except ImportError:
        print("  Warning: ollama package not installed. Run: .venv\\Scripts\\pip install ollama")
        return "regex"

    groups = _group_segments(segments)
    total_groups = len(groups)
    total_segs = len(segments)
    # 5d: seed with the last segment from the previous chunk if provided
    prev_combined = (prev_segment.get("text") or "").strip() if prev_segment else None

    try:
        for i, group in enumerate(groups):
            print(f"\r  Classifying group {i + 1}/{total_groups}...", end="", flush=True)
            combined = " ".join((s.get("text") or "").strip() for s in group).strip()

            # 5d: use context-aware prompt when a previous segment is available
            if prev_combined:
                prompt = _OLLAMA_PROMPT_WITH_CONTEXT.format(
                    prev_text=prev_combined.replace('"', "'"),
                    text=combined.replace('"', "'"),
                )
            else:
                prompt = _OLLAMA_PROMPT.format(text=combined.replace('"', "'"))

            response = ollama.chat(
                model=config.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response["message"]["content"].strip().lower()

            if "inquiry" in raw:
                label = "inquiry"
                confidence = 0.90 if raw.strip() in ("inquiry", "inquiry.") else 0.80
            elif "advocacy" in raw:
                label = "advocacy"
                confidence = 0.90 if raw.strip() in ("advocacy", "advocacy.") else 0.80
            else:
                # No recognisable label — fall back to regex for this group
                label, confidence = classify(combined)

            # 5b: override inquiry→advocacy when there is no ? and no invitation phrase
            label = _no_question_override(combined, label)
            # 5c: override advocacy→inquiry when there is a ?, question-start, or strong invitation
            label = _force_inquiry_override(combined, label)

            for seg in group:
                seg["label"] = label
                seg["confidence"] = confidence

            prev_combined = combined  # 5d: carry forward for next iteration

        print(f"\r  Classified {total_segs} segments ({total_groups} groups) with Ollama ({config.ollama_model}).  ")
        return "ollama"

    except ConnectionError:
        print(f"\n  Warning: Ollama is not running. Start it with: ollama serve")
        print(f"  Falling back to regex classifier.")
        return "regex"
    except Exception as e:
        print(f"\n  Warning: Ollama classification failed ({e}). Falling back to regex.")
        return "regex"
