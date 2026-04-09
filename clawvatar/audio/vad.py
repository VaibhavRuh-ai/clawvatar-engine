"""Voice Activity Detection using Silero VAD — detects when someone is speaking."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD requires exactly these chunk sizes
_CHUNK_SIZES = {16000: 512, 8000: 256}


class SileroVAD:
    """Silero Voice Activity Detection — lightweight, accurate, free."""

    def __init__(self, threshold: float = 0.3, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
        self._chunk_size = _CHUNK_SIZES.get(sample_rate, 512)

    def initialize(self) -> None:
        """Load Silero VAD model from torch hub."""
        import torch

        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model = model
        logger.info(f"Silero VAD initialized (chunk_size={self._chunk_size})")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech.

        Handles arbitrary chunk sizes by splitting/padding to the required 512 samples.
        """
        if self._model is None:
            raise RuntimeError("Not initialized. Call initialize() first.")

        import torch

        # Silero needs exactly 512 samples at 16kHz
        # Process the last 512 samples of the chunk (or pad if shorter)
        chunk = audio_chunk.flatten()
        if len(chunk) >= self._chunk_size:
            chunk = chunk[-self._chunk_size:]
        else:
            chunk = np.pad(chunk, (self._chunk_size - len(chunk), 0))

        tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        confidence = self._model(tensor, self.sample_rate).item()
        return confidence > self.threshold

    def get_confidence(self, audio_chunk: np.ndarray) -> float:
        """Get speech confidence score for an audio chunk."""
        if self._model is None:
            raise RuntimeError("Not initialized. Call initialize() first.")

        import torch

        chunk = audio_chunk.flatten()
        if len(chunk) >= self._chunk_size:
            chunk = chunk[-self._chunk_size:]
        else:
            chunk = np.pad(chunk, (self._chunk_size - len(chunk), 0))

        tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        return self._model(tensor, self.sample_rate).item()
