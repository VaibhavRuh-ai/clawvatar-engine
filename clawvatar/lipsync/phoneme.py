"""Text-to-phoneme-to-viseme pipeline using gruut.

Converts text directly to timed viseme sequences — no audio needed for detection.
When audio duration is known, distributes phoneme timing proportionally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# IPA phoneme → viseme mapping (covers English)
# Based on Disney/Pixar 10-viseme system
IPA_TO_VISEME: dict[str, str] = {
    # Vowels
    "ɑ": "A", "æ": "A", "ʌ": "A", "a": "A",     # open: pAt, fAther
    "i": "B", "ɪ": "B", "ɨ": "B",                  # smile: bEE
    "ɛ": "C", "e": "C",                              # mid open: bEd
    "ɔ": "D", "ɒ": "D", "o": "D",                   # round open: bOUght
    "u": "E", "ʊ": "E", "ʉ": "E",                   # tight round: blUE
    "ə": "X", "ɚ": "C",                              # schwa: weak
    # Diphthongs
    "aɪ": "A", "aʊ": "D", "ɔɪ": "D", "eɪ": "C", "oʊ": "D",
    # Consonants
    "p": "G", "b": "G", "m": "G",                    # bilabial: closed lips
    "f": "F", "v": "F",                               # labiodental: lip under teeth
    "θ": "F", "ð": "F",                               # dental: tongue tip
    "t": "G", "d": "G", "n": "H",                    # alveolar
    "s": "G", "z": "G",                               # sibilant
    "ʃ": "E", "ʒ": "E", "tʃ": "E", "dʒ": "E",     # postalveolar: pursed
    "k": "G", "ɡ": "G", "g": "G", "ŋ": "G",        # velar: back
    "l": "H",                                          # lateral
    "ɹ": "E", "r": "E",                               # approximant: pursed
    "w": "E",                                          # rounded
    "j": "B", "h": "X",                               # glottal
    "ʔ": "X",                                          # glottal stop
    # Stress/boundary markers (not visemes)
    "ˈ": None, "ˌ": None, "‖": None, "|": None,
}

# Average phoneme durations in seconds (rough estimates for natural speech)
PHONEME_DURATIONS: dict[str, float] = {
    # Vowels: longer
    "A": 0.08, "B": 0.07, "C": 0.07, "D": 0.08, "E": 0.07,
    # Consonants: shorter
    "F": 0.06, "G": 0.04, "H": 0.05,
    # Rest
    "X": 0.03, "REST": 0.05,
}


@dataclass
class VisemeEvent:
    """A single viseme with timing."""
    viseme: str
    start: float       # seconds from beginning
    duration: float    # seconds
    word: str          # the word this belongs to
    is_stressed: bool  # stressed syllable


@dataclass
class PauseEvent:
    """A pause between words/sentences."""
    start: float
    duration: float


def text_to_phonemes(text: str, lang: str = "en-us") -> list[dict]:
    """Convert text to phoneme list using gruut.

    Returns:
        List of {"word": str, "phonemes": list[str], "stressed": list[bool]}
    """
    from gruut import sentences

    result = []
    for sent in sentences(text, lang=lang):
        for word in sent:
            if not word.phonemes:
                if word.text in (".", "!", "?", ",", ";", ":"):
                    result.append({"word": word.text, "phonemes": ["‖"], "stressed": [False]})
                continue
            stressed = []
            for p in word.phonemes:
                stressed.append("ˈ" in p or "ˌ" in p)
            result.append({
                "word": word.text,
                "phonemes": word.phonemes,
                "stressed": stressed,
            })
    return result


def phonemes_to_visemes(phoneme_data: list[dict]) -> list[VisemeEvent]:
    """Convert phoneme list to viseme sequence with estimated durations."""
    visemes = []

    for entry in phoneme_data:
        word = entry["word"]
        for i, phoneme_str in enumerate(entry["phonemes"]):
            is_stressed = entry["stressed"][i] if i < len(entry["stressed"]) else False

            # Clean phoneme (remove stress marks)
            clean = phoneme_str.replace("ˈ", "").replace("ˌ", "").strip()
            if not clean:
                continue

            # Map IPA → viseme
            viseme = IPA_TO_VISEME.get(clean)
            if viseme is None:
                # Try single char match for multi-char phonemes
                for char in clean:
                    viseme = IPA_TO_VISEME.get(char)
                    if viseme:
                        break
            if viseme is None:
                viseme = "X"

            # Duration
            dur = PHONEME_DURATIONS.get(viseme, 0.05)
            if is_stressed:
                dur *= 1.3  # stressed syllables are longer

            visemes.append(VisemeEvent(
                viseme=viseme,
                start=0,  # will be computed in build_timeline
                duration=dur,
                word=word,
                is_stressed=is_stressed,
            ))

    return visemes


def build_timeline(
    visemes: list[VisemeEvent],
    total_duration: float,
    sentence_pause: float = 0.15,
    word_gap: float = 0.02,
) -> list[VisemeEvent]:
    """Distribute viseme events across the total audio duration.

    Scales phoneme durations proportionally to fit the actual audio length.
    """
    if not visemes:
        return []

    # Calculate total estimated duration
    estimated_total = sum(v.duration for v in visemes)
    if estimated_total <= 0:
        return visemes

    # Scale factor to fit actual audio duration (leave 5% margin)
    scale = (total_duration * 0.95) / estimated_total

    # Assign timestamps
    cursor = total_duration * 0.025  # small lead-in
    prev_word = ""
    for v in visemes:
        # Add gap between words
        if v.word != prev_word and prev_word:
            if v.word in (".", "!", "?"):
                cursor += sentence_pause * scale
            else:
                cursor += word_gap * scale
        prev_word = v.word

        v.start = cursor
        v.duration = v.duration * scale
        cursor += v.duration

    return visemes
