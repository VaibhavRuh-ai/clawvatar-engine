"""Expression engine — maps audio prosody to lifelike avatar expressions.

Drives eyebrows, eyes, head movement, and emotional expressions from
real-time prosody analysis. No AI model needed — pure signal-to-expression mapping.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from clawvatar.audio.prosody import ProsodyFeatures


class ExpressionState:
    """Current avatar expression state — all values 0.0 to 1.0 unless noted."""

    def __init__(self):
        # Mouth (driven by viseme system, we add modulation)
        self.mouth_intensity: float = 1.0  # scale factor for mouth shapes

        # Eyebrows
        self.brow_raise_l: float = 0.0
        self.brow_raise_r: float = 0.0
        self.brow_inner_up: float = 0.0
        self.brow_down_l: float = 0.0
        self.brow_down_r: float = 0.0

        # Eyes
        self.eye_wide_l: float = 0.0
        self.eye_wide_r: float = 0.0
        self.eye_squint_l: float = 0.0
        self.eye_squint_r: float = 0.0
        self.eye_look_x: float = 0.0  # -1 left, +1 right
        self.eye_look_y: float = 0.0  # -1 down, +1 up
        self.blink_l: float = 0.0
        self.blink_r: float = 0.0

        # Emotions (VRM presets)
        self.happy: float = 0.0
        self.angry: float = 0.0
        self.sad: float = 0.0
        self.surprised: float = 0.0
        self.relaxed: float = 0.0

        # Head (degrees)
        self.head_nod: float = 0.0  # pitch offset from speech
        self.head_tilt: float = 0.0  # roll offset from speech
        self.head_turn: float = 0.0  # yaw offset from speech


class ExpressionEngine:
    """Maps prosody features to avatar expressions in real-time."""

    def __init__(self):
        self._state = ExpressionState()
        self._target = ExpressionState()

        # Smoothing (per-second decay rates)
        self._smooth_fast = 0.3   # fast-moving features (mouth, nods)
        self._smooth_med = 0.15   # medium (eyebrows, eyes)
        self._smooth_slow = 0.08  # slow (emotions, eye gaze)

        # Idle eye movement
        self._next_eye_shift = time.time() + random.uniform(1, 3)
        self._eye_target_x = 0.0
        self._eye_target_y = 0.0

        # Head nod rhythm
        self._nod_phase = 0.0

        # Blink
        self._last_blink = time.time()
        self._next_blink = self._random_blink_time()
        self._blink_progress = 1.0  # 1.0 = not blinking

    def _random_blink_time(self) -> float:
        return random.uniform(2.5, 5.0)

    def update(self, prosody: Optional[ProsodyFeatures], dt: float = 1 / 30) -> ExpressionState:
        """Update expression state from prosody features.

        Args:
            prosody: Current prosody analysis (None if silent).
            dt: Time delta.

        Returns:
            Current expression state.
        """
        t = self._target
        now = time.time()

        if prosody and not prosody.is_pause and prosody.is_voiced:
            self._update_speaking(t, prosody, dt)
        else:
            self._update_idle(t, prosody, dt)

        # --- Blink (always active) ---
        time_since_blink = now - self._last_blink
        if time_since_blink >= self._next_blink:
            self._blink_progress = 0.0
            self._last_blink = now
            self._next_blink = self._random_blink_time()

        if self._blink_progress < 1.0:
            self._blink_progress += dt * 8  # ~125ms blink
            blink_val = math.sin(min(self._blink_progress, 1.0) * math.pi)
            t.blink_l = blink_val
            t.blink_r = blink_val
        else:
            t.blink_l = 0.0
            t.blink_r = 0.0

        # --- Idle eye movement ---
        if now >= self._next_eye_shift:
            self._eye_target_x = random.uniform(-0.3, 0.3)
            self._eye_target_y = random.uniform(-0.15, 0.15)
            self._next_eye_shift = now + random.uniform(1.5, 4.0)
        t.eye_look_x = self._eye_target_x
        t.eye_look_y = self._eye_target_y

        # --- Smooth all values toward target ---
        self._smooth(dt)

        return self._state

    def _update_speaking(self, t: ExpressionState, p: ProsodyFeatures, dt: float) -> None:
        """Update expression targets while speaking."""

        # --- Mouth intensity scales with energy ---
        t.mouth_intensity = 0.6 + min(0.4, p.rms * 3)

        # --- Head nods on syllable beats ---
        if p.syllable_beat:
            self._nod_phase = 1.0
        if self._nod_phase > 0:
            t.head_nod = -math.sin(self._nod_phase * math.pi) * 3.0  # degrees
            self._nod_phase -= dt * 5
            if self._nod_phase < 0:
                self._nod_phase = 0

        # --- Eyebrow raise on pitch peaks ---
        if p.pitch_delta > 15:
            intensity = min(1.0, p.pitch_delta / 60)
            t.brow_raise_l = intensity * 0.6
            t.brow_raise_r = intensity * 0.6
            t.brow_inner_up = intensity * 0.4
        elif p.pitch_delta < -15:
            t.brow_down_l = min(0.3, abs(p.pitch_delta) / 80)
            t.brow_down_r = min(0.3, abs(p.pitch_delta) / 80)
        else:
            t.brow_raise_l *= 0.8
            t.brow_raise_r *= 0.8
            t.brow_inner_up *= 0.8
            t.brow_down_l *= 0.8
            t.brow_down_r *= 0.8

        # --- Eye wideness on emphasis ---
        if p.energy_delta > 0.06:
            t.eye_wide_l = min(0.4, p.energy_delta * 3)
            t.eye_wide_r = min(0.4, p.energy_delta * 3)
        else:
            t.eye_wide_l *= 0.85
            t.eye_wide_r *= 0.85

        # --- Head tilt on questions (rising pitch trend) ---
        if p.emotion == "question":
            t.head_tilt = 3.0 * p.emotion_intensity  # slight tilt
            t.brow_raise_l = max(t.brow_raise_l, 0.4)
            t.brow_raise_r = max(t.brow_raise_r, 0.4)
        else:
            t.head_tilt *= 0.9

        # --- Subtle head turn following speech rhythm ---
        t.head_turn = math.sin(time.time() * 0.8) * 2.0 * min(1.0, p.rms * 5)

        # --- Emotion expressions ---
        self._apply_emotion(t, p)

        # --- Eye squint during smile ---
        if t.happy > 0.3:
            t.eye_squint_l = t.happy * 0.5
            t.eye_squint_r = t.happy * 0.5

        # --- Faster blinks during animated speech ---
        if p.speech_rate > 4:
            self._next_blink = min(self._next_blink, 2.0)

    def _update_idle(self, t: ExpressionState, p: Optional[ProsodyFeatures], dt: float) -> None:
        """Update expression targets while idle/silent."""
        # Relax everything toward neutral
        t.mouth_intensity = 1.0
        t.brow_raise_l *= 0.92
        t.brow_raise_r *= 0.92
        t.brow_inner_up *= 0.92
        t.brow_down_l *= 0.92
        t.brow_down_r *= 0.92
        t.eye_wide_l *= 0.9
        t.eye_wide_r *= 0.9
        t.eye_squint_l *= 0.9
        t.eye_squint_r *= 0.9
        t.head_nod *= 0.9
        t.head_tilt *= 0.9
        t.head_turn *= 0.9

        # Emotions decay
        t.happy *= 0.95
        t.angry *= 0.95
        t.sad *= 0.95
        t.surprised *= 0.95
        t.relaxed = 0.15  # slight resting smile

        # Gentle breathing movement
        breath = math.sin(time.time() * 0.4 * math.pi * 2) * 0.3
        t.head_nod = breath

    def _apply_emotion(self, t: ExpressionState, p: ProsodyFeatures) -> None:
        """Map detected emotion to VRM expression weights."""
        intensity = p.emotion_intensity

        # Decay all emotions first
        t.happy *= 0.9
        t.angry *= 0.9
        t.sad *= 0.9
        t.surprised *= 0.9
        t.relaxed *= 0.9

        if p.emotion == "excited" or p.emotion == "happy":
            t.happy = max(t.happy, intensity * 0.6)
        elif p.emotion == "calm":
            t.relaxed = max(t.relaxed, intensity * 0.4)
        elif p.emotion == "question":
            t.surprised = max(t.surprised, intensity * 0.3)
        elif p.emotion == "emphasis":
            # Brief intensity boost, no specific emotion
            pass

    def _smooth(self, dt: float) -> None:
        """Smooth state values toward targets."""
        s = self._state
        t = self._target

        def lerp(current: float, target: float, rate: float) -> float:
            return current + (target - current) * min(1.0, rate)

        fast = self._smooth_fast
        med = self._smooth_med
        slow = self._smooth_slow

        # Fast
        s.mouth_intensity = lerp(s.mouth_intensity, t.mouth_intensity, fast)
        s.head_nod = lerp(s.head_nod, t.head_nod, fast)
        s.head_tilt = lerp(s.head_tilt, t.head_tilt, fast)
        s.head_turn = lerp(s.head_turn, t.head_turn, fast)
        s.blink_l = lerp(s.blink_l, t.blink_l, 0.5)  # blinks are quick
        s.blink_r = lerp(s.blink_r, t.blink_r, 0.5)

        # Medium
        s.brow_raise_l = lerp(s.brow_raise_l, t.brow_raise_l, med)
        s.brow_raise_r = lerp(s.brow_raise_r, t.brow_raise_r, med)
        s.brow_inner_up = lerp(s.brow_inner_up, t.brow_inner_up, med)
        s.brow_down_l = lerp(s.brow_down_l, t.brow_down_l, med)
        s.brow_down_r = lerp(s.brow_down_r, t.brow_down_r, med)
        s.eye_wide_l = lerp(s.eye_wide_l, t.eye_wide_l, med)
        s.eye_wide_r = lerp(s.eye_wide_r, t.eye_wide_r, med)
        s.eye_squint_l = lerp(s.eye_squint_l, t.eye_squint_l, med)
        s.eye_squint_r = lerp(s.eye_squint_r, t.eye_squint_r, med)

        # Slow
        s.eye_look_x = lerp(s.eye_look_x, t.eye_look_x, slow)
        s.eye_look_y = lerp(s.eye_look_y, t.eye_look_y, slow)
        s.happy = lerp(s.happy, t.happy, slow)
        s.angry = lerp(s.angry, t.angry, slow)
        s.sad = lerp(s.sad, t.sad, slow)
        s.surprised = lerp(s.surprised, t.surprised, slow)
        s.relaxed = lerp(s.relaxed, t.relaxed, slow)

    def to_vrm_weights(self, state: ExpressionState) -> dict[str, float]:
        """Convert expression state to VRM expression weights dict.

        Returns dict with VRM expression names and blend shape values.
        """
        w = {}

        # Emotions (VRM preset expressions)
        if state.happy > 0.01:
            w["happy"] = round(min(1.0, state.happy), 4)
        if state.angry > 0.01:
            w["angry"] = round(min(1.0, state.angry), 4)
        if state.sad > 0.01:
            w["relaxed"] = round(min(1.0, state.sad), 4)
        if state.surprised > 0.01:
            w["surprised"] = round(min(1.0, state.surprised), 4)
        if state.relaxed > 0.01:
            w["relaxed"] = round(max(w.get("relaxed", 0), min(1.0, state.relaxed)), 4)

        # Blink
        if state.blink_l > 0.01:
            w["blinkLeft"] = round(min(1.0, state.blink_l), 4)
        if state.blink_r > 0.01:
            w["blinkRight"] = round(min(1.0, state.blink_r), 4)

        return w

    def to_head_pose(self, state: ExpressionState) -> dict[str, float]:
        """Get head pose offsets from expression state.

        Returns dict with yaw, pitch, roll in degrees.
        """
        return {
            "yaw": round(state.head_turn, 3),
            "pitch": round(state.head_nod, 3),
            "roll": round(state.head_tilt, 3),
        }

    def get_mouth_intensity(self) -> float:
        """Get current mouth shape intensity multiplier."""
        return self._state.mouth_intensity

    def get_eye_look(self) -> dict[str, float]:
        """Get eye look direction."""
        return {
            "x": round(self._state.eye_look_x, 3),
            "y": round(self._state.eye_look_y, 3),
        }
