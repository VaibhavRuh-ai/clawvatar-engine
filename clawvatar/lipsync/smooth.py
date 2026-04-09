"""Smooth lip-sync processor — adds coarticulation, energy scaling, and natural timing.

Takes raw viseme codes from the detector and produces smooth, realistic
blend shape weights with proper transitions.
"""

from __future__ import annotations

import math
import time
from collections import deque

from clawvatar.lipsync.visemes import VISEME_TO_VRM, get_blendshape_weights


# Coarticulation: how much the next viseme bleeds into the current one
COARTIC_FACTOR = 0.25

# Ease curve for viseme transitions (faster open, slower close)
def ease_out_quad(t: float) -> float:
    """Quick start, gradual end — natural mouth opening."""
    return t * (2 - t)

def ease_in_quad(t: float) -> float:
    """Gradual start, quick end — natural mouth closing."""
    return t * t


class SmoothLipSync:
    """Produces smooth, realistic lip-sync weights from viseme codes + audio energy."""

    def __init__(
        self,
        transition_speed: float = 12.0,  # visemes per second
        energy_influence: float = 0.6,   # how much energy affects mouth size
        coarticulation: float = 0.25,    # next-viseme bleed
        format: str = "vrm",
    ):
        self.transition_speed = transition_speed
        self.energy_influence = energy_influence
        self.coarticulation = coarticulation
        self.format = format

        # State
        self._current_weights: dict[str, float] = {}
        self._target_weights: dict[str, float] = {}
        self._prev_viseme: str = "REST"
        self._current_viseme: str = "REST"
        self._transition_progress: float = 1.0  # 0=start, 1=complete
        self._viseme_history: deque[str] = deque(maxlen=5)
        self._energy_smooth: float = 0.0
        self._jaw_phase: float = 0.0  # for jaw oscillation

    def update(
        self,
        viseme: str | None,
        energy: float,
        is_speaking: bool,
        dt: float = 1 / 30,
    ) -> dict[str, float]:
        """Update and return smooth blend shape weights.

        Args:
            viseme: Current detected viseme code, or None if silent.
            energy: Audio RMS energy (0-1 range).
            is_speaking: Whether speech is detected.
            dt: Time delta.

        Returns:
            Smooth blend shape weights dict.
        """
        # Smooth energy
        self._energy_smooth += (energy - self._energy_smooth) * 0.3

        if not is_speaking or viseme is None:
            viseme = "REST"

        # New viseme detected
        if viseme != self._current_viseme:
            self._prev_viseme = self._current_viseme
            self._current_viseme = viseme
            self._transition_progress = 0.0
            self._viseme_history.append(viseme)

        # Advance transition
        if self._transition_progress < 1.0:
            self._transition_progress += dt * self.transition_speed
            self._transition_progress = min(1.0, self._transition_progress)

        # Get base weights for prev and current viseme
        prev_w = get_blendshape_weights(self._prev_viseme, self.format)
        curr_w = get_blendshape_weights(self._current_viseme, self.format)

        # Eased transition
        if self._current_viseme == "REST" or self._current_viseme == "X":
            # Closing mouth — ease in (gradual)
            t = ease_in_quad(self._transition_progress)
        else:
            # Opening mouth — ease out (quick)
            t = ease_out_quad(self._transition_progress)

        # Blend prev → current
        all_keys = set(prev_w.keys()) | set(curr_w.keys())
        blended = {}
        for k in all_keys:
            p = prev_w.get(k, 0.0)
            c = curr_w.get(k, 0.0)
            blended[k] = p + (c - p) * t

        # Energy scaling — scale mouth shapes by audio energy
        if is_speaking and self._energy_smooth > 0.01:
            # Base intensity + energy modulation
            intensity = (1.0 - self.energy_influence) + self.energy_influence * min(1.0, self._energy_smooth * 5)
            for k in blended:
                if "mouth" in k or "jaw" in k:
                    blended[k] *= intensity

        # Jaw oscillation during speech — subtle rhythmic jaw movement
        if is_speaking and self._energy_smooth > 0.02:
            self._jaw_phase += dt * 15  # ~15Hz oscillation (natural jaw frequency)
            jaw_osc = math.sin(self._jaw_phase) * 0.05 * min(1.0, self._energy_smooth * 3)
            # Add to any mouth_a shape
            mouth_key = "blendShape2.mouth_a" if self.format == "vrm" else "jawOpen"
            blended[mouth_key] = max(blended.get(mouth_key, 0.0), abs(jaw_osc))

        # Coarticulation: blend a hint of the previous viseme
        if self.coarticulation > 0 and self._transition_progress < 0.7:
            for k, v in prev_w.items():
                if k in blended:
                    blended[k] += v * self.coarticulation * (1.0 - self._transition_progress)

        # Clamp and filter
        result = {}
        for k, v in blended.items():
            v = max(0.0, min(1.0, v))
            if v > 0.005:
                result[k] = round(v, 4)

        self._current_weights = result
        return result

    def reset(self) -> None:
        self._current_weights = {}
        self._target_weights = {}
        self._prev_viseme = "REST"
        self._current_viseme = "REST"
        self._transition_progress = 1.0
        self._energy_smooth = 0.0
        self._jaw_phase = 0.0
