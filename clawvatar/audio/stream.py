"""Audio stream buffer — accumulates audio chunks and provides them to the pipeline."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioStreamBuffer:
    """Ring buffer for incoming audio chunks with overlap support."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 40,
        context_chunks: int = 5,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.context_chunks = context_chunks
        self._buffer: deque[np.ndarray] = deque(maxlen=context_chunks + 1)
        self._is_speaking = False

    def push(self, audio: np.ndarray) -> None:
        """Add an audio chunk to the buffer."""
        self._buffer.append(audio)

    def get_context_window(self) -> Optional[np.ndarray]:
        """Get current audio context (recent chunks concatenated).

        Returns:
            Concatenated audio array with context, or None if buffer empty.
        """
        if not self._buffer:
            return None
        return np.concatenate(list(self._buffer))

    def get_latest_chunk(self) -> Optional[np.ndarray]:
        """Get the most recently pushed chunk."""
        if not self._buffer:
            return None
        return self._buffer[-1]

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @is_speaking.setter
    def is_speaking(self, value: bool) -> None:
        self._is_speaking = value

    def clear(self) -> None:
        self._buffer.clear()
        self._is_speaking = False
