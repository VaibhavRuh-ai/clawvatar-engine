"""Energy-based viseme detection — simple fallback, no dependencies.

Maps audio energy and spectral features to visemes. Not as accurate as Rhubarb
but works everywhere without external binaries.
"""

from __future__ import annotations

import logging

import numpy as np

from clawvatar.lipsync.visemes import get_blendshape_weights
from clawvatar.providers.base import LipSyncProvider

logger = logging.getLogger(__name__)


class EnergyLipSync(LipSyncProvider):
    """Audio energy + spectral analysis → viseme detection. No external deps."""

    def __init__(self):
        self._prev_energy = 0.0

    def initialize(self) -> None:
        logger.info("Energy-based lip-sync initialized (no external dependencies)")

    def detect_viseme(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        rms = np.sqrt(np.mean(audio_chunk ** 2))

        if rms < 0.008:
            return "X"  # Silence / rest

        # Simple spectral analysis
        fft = np.abs(np.fft.rfft(audio_chunk))
        freqs = np.fft.rfftfreq(len(audio_chunk), 1 / sample_rate)

        # Energy in frequency bands
        low = np.mean(fft[(freqs >= 80) & (freqs < 400)])    # fundamental
        mid = np.mean(fft[(freqs >= 400) & (freqs < 2000)])   # formants
        high = np.mean(fft[(freqs >= 2000) & (freqs < 6000)]) # fricatives

        total = low + mid + high + 1e-8

        low_ratio = low / total
        mid_ratio = mid / total
        high_ratio = high / total

        # Map spectral shape to visemes
        if high_ratio > 0.4:
            # High frequency dominant → fricatives (s, f, th)
            return "F" if mid_ratio > 0.2 else "G"
        elif low_ratio > 0.5 and rms > 0.15:
            # Low frequency dominant, loud → open vowels
            return "A" if rms > 0.25 else "C"
        elif mid_ratio > 0.4:
            # Mid frequency dominant → front vowels
            if rms > 0.15:
                return "D"  # "oh" — rounder
            return "B"  # "ee" — wider
        elif rms > 0.05:
            return "E" if low_ratio > 0.4 else "C"
        else:
            return "X"

    def detect_viseme_weights(
        self, audio_chunk: np.ndarray, sample_rate: int = 16000
    ) -> dict[str, float]:
        viseme = self.detect_viseme(audio_chunk, sample_rate)

        # Scale blend shape weights by audio energy for natural feel
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        intensity = min(1.0, rms / 0.3)

        weights = get_blendshape_weights(viseme)
        return {k: v * intensity for k, v in weights.items()}
