"""Expression planner — analyzes agent text to plan emotions, emphasis, and gestures.

Rule-based instant analysis. Optional LLM (Gemini Flash) for deeper understanding.
Outputs a timeline of expression events that the animation builder merges with visemes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExpressionEvent:
    """A planned expression change at a specific time."""
    start: float           # seconds from beginning
    duration: float        # how long the expression lasts
    emotion: str           # "happy", "sad", "excited", "calm", "empathetic", "neutral"
    intensity: float       # 0.0 to 1.0
    head_gesture: str      # "nod", "tilt", "shake", "none"
    eyebrow: str           # "raise", "furrow", "flash", "none"
    emphasis_words: list[str] = field(default_factory=list)


@dataclass
class ExpressionPlan:
    """Full expression plan for an utterance."""
    events: list[ExpressionEvent]
    overall_emotion: str
    overall_intensity: float


# Emotion keywords
HAPPY_WORDS = {"happy", "great", "wonderful", "amazing", "excellent", "love", "glad",
               "awesome", "fantastic", "perfect", "beautiful", "good", "nice", "thank",
               "welcome", "pleasure", "enjoy", "exciting", "brilliant", "terrific"}

SAD_WORDS = {"sorry", "unfortunately", "sad", "bad", "terrible", "awful", "regret",
             "disappointing", "upset", "worried", "concern", "problem", "issue", "fail",
             "difficult", "struggle", "trouble", "pain", "loss", "miss"}

EXCITED_WORDS = {"wow", "incredible", "unbelievable", "absolutely", "definitely",
                 "totally", "really", "extremely", "super", "very"}

EMPATHETIC_WORDS = {"understand", "feel", "hear", "imagine", "must be", "that sounds",
                    "i see", "of course", "certainly"}

THINKING_WORDS = {"well", "hmm", "let me", "think", "consider", "perhaps", "maybe",
                  "possibly"}


def plan_expressions(
    text: str,
    audio_duration: float,
) -> ExpressionPlan:
    """Analyze text and create expression plan.

    Args:
        text: The agent's text to speak.
        audio_duration: Total duration of TTS audio in seconds.

    Returns:
        ExpressionPlan with timed events.
    """
    text_lower = text.lower()
    sentences = _split_sentences(text)
    events = []

    if not sentences:
        return ExpressionPlan(events=[], overall_emotion="neutral", overall_intensity=0.3)

    # Time per sentence (proportional to word count)
    total_words = sum(len(s.split()) for s in sentences)
    if total_words == 0:
        total_words = 1

    cursor = 0.0
    overall_emotion = "neutral"
    overall_intensity = 0.3

    for sent in sentences:
        words = sent.split()
        sent_duration = (len(words) / total_words) * audio_duration
        sent_lower = sent.lower()

        # Detect emotion for this sentence
        emotion, intensity = _detect_sentence_emotion(sent_lower)
        if intensity > overall_intensity:
            overall_emotion = emotion
            overall_intensity = intensity

        # Detect emphasis words
        emphasis = _find_emphasis_words(sent)

        # Detect head gesture
        gesture = "none"
        if sent.strip().endswith("?"):
            gesture = "tilt"
        elif sent.strip().endswith("!"):
            gesture = "nod"
        elif any(w in sent_lower for w in ["yes", "right", "exactly", "agree", "sure"]):
            gesture = "nod"
        elif any(w in sent_lower for w in ["no", "not", "never", "don't", "can't"]):
            gesture = "shake"

        # Detect eyebrow
        eyebrow = "none"
        if "?" in sent:
            eyebrow = "raise"
        elif "!" in sent:
            eyebrow = "flash"
        elif emotion in ("sad", "empathetic"):
            eyebrow = "furrow"
        elif emphasis:
            eyebrow = "flash"

        events.append(ExpressionEvent(
            start=cursor,
            duration=sent_duration,
            emotion=emotion,
            intensity=intensity,
            head_gesture=gesture,
            eyebrow=eyebrow,
            emphasis_words=emphasis,
        ))

        cursor += sent_duration

    return ExpressionPlan(
        events=events,
        overall_emotion=overall_emotion,
        overall_intensity=overall_intensity,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_sentence_emotion(sent: str) -> tuple[str, float]:
    """Detect primary emotion and intensity for a sentence."""
    words = set(sent.split())

    scores = {
        "happy": len(words & HAPPY_WORDS) * 0.3,
        "sad": len(words & SAD_WORDS) * 0.3,
        "excited": len(words & EXCITED_WORDS) * 0.25,
        "empathetic": len(words & EMPATHETIC_WORDS) * 0.25,
        "thinking": len(words & THINKING_WORDS) * 0.2,
    }

    # Punctuation boost
    if "!" in sent:
        scores["excited"] += 0.2
        scores["happy"] += 0.1
    if "?" in sent:
        scores["thinking"] += 0.15

    # Exclamation count
    scores["excited"] += sent.count("!") * 0.1

    best = max(scores, key=scores.get)
    intensity = min(1.0, scores[best])

    if intensity < 0.1:
        return "neutral", 0.3

    return best, min(1.0, 0.3 + intensity)


def _find_emphasis_words(sent: str) -> list[str]:
    """Find words that should be emphasized (caps, important words)."""
    emphasis = []
    for word in sent.split():
        clean = word.strip(".,!?;:'\"")
        if not clean:
            continue
        # ALL CAPS words
        if clean.isupper() and len(clean) > 1:
            emphasis.append(clean)
        # Words after "very", "really", "so", "extremely"
        # Key adjectives/adverbs
        if clean.lower() in EXCITED_WORDS | HAPPY_WORDS:
            emphasis.append(clean)
    return emphasis


def expression_to_vrm_weights(emotion: str, intensity: float) -> dict[str, float]:
    """Convert an emotion to VRM expression weights."""
    weights = {}

    if emotion == "happy" or emotion == "excited":
        weights["happy"] = intensity * 0.6
    elif emotion == "sad":
        weights["relaxed"] = intensity * 0.4  # VRM "relaxed" looks slightly sad
    elif emotion == "empathetic":
        weights["relaxed"] = intensity * 0.3
    elif emotion == "thinking":
        # Subtle — no strong emotion
        pass

    return weights
