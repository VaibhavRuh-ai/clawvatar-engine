"""Base provider interfaces for the 3D avatar engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LipSyncProvider(ABC):
    """Detects visemes from audio for lip-sync animation."""

    @abstractmethod
    def initialize(self) -> None:
        """Load models or validate dependencies."""

    @abstractmethod
    def detect_viseme(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """Detect current viseme from an audio chunk.

        Returns:
            Viseme code string: "REST", "A", "E", "I", "O", "U",
            "FV", "MBP", "WR", "LNT", "SZ", "TH", "CH", "K", "R"
        """

    @abstractmethod
    def detect_viseme_weights(
        self, audio_chunk: np.ndarray, sample_rate: int = 16000
    ) -> dict[str, float]:
        """Detect viseme blend weights (for smooth transitions).

        Returns:
            Dict mapping viseme codes to weights (0.0-1.0).
        """
