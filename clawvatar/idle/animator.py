"""Idle animations — blinking, subtle head movement, breathing when avatar is not speaking."""

from __future__ import annotations

import math
import random
import time

import numpy as np


class IdleAnimator:
    """Generates natural idle motion parameters when the avatar is not speaking."""

    def __init__(
        self,
        blink_interval: float = 3.5,
        blink_duration: float = 0.15,
        movement_scale: float = 0.3,
        breathing_rate: float = 0.25,  # breaths per second
    ):
        self.blink_interval = blink_interval
        self.blink_duration = blink_duration
        self.movement_scale = movement_scale
        self.breathing_rate = breathing_rate

        self._last_blink_time = time.time()
        self._next_blink_delay = self._random_blink_delay()
        self._start_time = time.time()

    def _random_blink_delay(self) -> float:
        """Randomize blink interval for natural feel."""
        return self.blink_interval + random.uniform(-1.0, 1.5)

    def get_idle_params(self) -> dict:
        """Get current idle animation parameters.

        Returns:
            Dict with:
                - blink: float 0-1 (0=open, 1=closed)
                - head_yaw: float in degrees
                - head_pitch: float in degrees
                - breathing: float 0-1
        """
        now = time.time()
        elapsed = now - self._start_time

        # Blink
        blink = 0.0
        time_since_blink = now - self._last_blink_time
        if time_since_blink >= self._next_blink_delay:
            blink_progress = (time_since_blink - self._next_blink_delay) / self.blink_duration
            if blink_progress < 1.0:
                # Smooth blink curve: quick close, slower open
                blink = math.sin(blink_progress * math.pi)
            else:
                self._last_blink_time = now
                self._next_blink_delay = self._random_blink_delay()

        # Subtle head movement (Perlin-like using layered sinusoids)
        head_yaw = (
            math.sin(elapsed * 0.3) * 1.5
            + math.sin(elapsed * 0.7 + 1.2) * 0.8
        ) * self.movement_scale

        head_pitch = (
            math.sin(elapsed * 0.2 + 0.5) * 1.0
            + math.sin(elapsed * 0.5 + 2.1) * 0.5
        ) * self.movement_scale

        # Breathing — gentle vertical oscillation
        breathing = (math.sin(elapsed * self.breathing_rate * 2 * math.pi) + 1) / 2

        return {
            "blink": blink,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "head_roll": math.sin(elapsed * 0.15) * 0.3 * self.movement_scale,
            "breathing": breathing,
        }

    def reset(self) -> None:
        """Reset animation state."""
        self._start_time = time.time()
        self._last_blink_time = time.time()
        self._next_blink_delay = self._random_blink_delay()
