"""Blend shape animator — combines viseme lip-sync + idle animations into final weights.

Handles smooth interpolation between visemes and layering of idle animations
(blink, breathing, head movement) on top of lip-sync weights.
Auto-detects ARKit vs VRM blend shape format.
"""

from __future__ import annotations

import time

from clawvatar.idle.animator import IdleAnimator
from clawvatar.lipsync.visemes import get_blendshape_weights, interpolate_weights


class BlendShapeAnimator:
    """Combines all animation sources into final blend shape weights per frame."""

    def __init__(
        self,
        smoothing: float = 0.3,
        min_hold_ms: int = 60,
        idle_blink_interval: float = 3.5,
        idle_movement_scale: float = 0.3,
        blendshape_format: str = "vrm",
    ):
        self.smoothing = smoothing
        self.min_hold_ms = min_hold_ms
        self.blendshape_format = blendshape_format

        self.idle = IdleAnimator(
            blink_interval=idle_blink_interval,
            movement_scale=idle_movement_scale,
        )

        # State
        self._current_weights: dict[str, float] = {}
        self._current_viseme: str = "REST"
        self._viseme_start_time: float = 0.0
        self._head_yaw: float = 0.0
        self._head_pitch: float = 0.0
        self._head_roll: float = 0.0

    def update(
        self,
        viseme: str | None = None,
        viseme_weights: dict[str, float] | None = None,
        is_speaking: bool = False,
        dt: float = 1 / 30,
    ) -> dict:
        """Update animation state and return final blend shape weights + head pose."""
        now = time.time()
        fmt = self.blendshape_format

        # --- Lip-sync weights ---
        if viseme_weights is not None:
            target_lip = viseme_weights
        elif viseme is not None and viseme != self._current_viseme:
            elapsed_ms = (now - self._viseme_start_time) * 1000
            if elapsed_ms >= self.min_hold_ms:
                self._current_viseme = viseme
                self._viseme_start_time = now
                target_lip = get_blendshape_weights(viseme, fmt)
            else:
                target_lip = get_blendshape_weights(self._current_viseme, fmt)
        else:
            target_lip = get_blendshape_weights(self._current_viseme, fmt)

        # Smooth interpolation
        lerp_factor = 1.0 - self.smoothing
        self._current_weights = interpolate_weights(
            self._current_weights, target_lip, lerp_factor
        )

        # --- Idle animations ---
        idle = self.idle.get_idle_params()

        final_weights = dict(self._current_weights)
        blink = idle["blink"]

        # Layer blink — use correct key names per format
        if blink > 0.01:
            if fmt == "vrm":
                final_weights["blendShape2.Blink_L"] = max(
                    final_weights.get("blendShape2.Blink_L", 0.0), blink
                )
                final_weights["blendShape2.Blink_R"] = max(
                    final_weights.get("blendShape2.Blink_R", 0.0), blink
                )
            else:
                final_weights["eyeBlinkLeft"] = max(
                    final_weights.get("eyeBlinkLeft", 0.0), blink
                )
                final_weights["eyeBlinkRight"] = max(
                    final_weights.get("eyeBlinkRight", 0.0), blink
                )

        # Head pose
        idle_scale = 0.3 if is_speaking else 1.0
        target_yaw = idle["head_yaw"] * idle_scale
        target_pitch = idle["head_pitch"] * idle_scale
        target_roll = idle["head_roll"] * idle_scale

        self._head_yaw += (target_yaw - self._head_yaw) * 0.1
        self._head_pitch += (target_pitch - self._head_pitch) * 0.1
        self._head_roll += (target_roll - self._head_roll) * 0.1

        # Clean up near-zero weights
        final_weights = {k: v for k, v in final_weights.items() if abs(v) > 0.001}

        return {
            "blend_shapes": final_weights,
            "head_yaw": self._head_yaw,
            "head_pitch": self._head_pitch,
            "head_roll": self._head_roll,
        }

    def reset(self) -> None:
        self._current_weights = {}
        self._current_viseme = "REST"
        self._head_yaw = 0.0
        self._head_pitch = 0.0
        self._head_roll = 0.0
        self.idle.reset()
