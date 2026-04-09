"""Animation timeline builder — merges phoneme visemes + expression plan into frame-by-frame weights.

Pre-computes the entire animation before playback. Each frame has:
- Mouth blend shapes (from viseme timeline)
- Expression weights (from expression plan)
- Head pose (gestures + idle)
- Eye blink timing

Output: list of frame dicts, one per frame at target FPS.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from clawvatar.animation.expression_planner import ExpressionEvent, ExpressionPlan, expression_to_vrm_weights
from clawvatar.lipsync.phoneme import VisemeEvent
from clawvatar.lipsync.visemes import VISEME_TO_VRM


@dataclass
class AnimationFrame:
    """One frame of pre-computed animation."""
    time: float
    weights: dict[str, float]
    head_yaw: float
    head_pitch: float
    head_roll: float


def build_animation_timeline(
    visemes: list[VisemeEvent],
    expression_plan: ExpressionPlan,
    total_duration: float,
    fps: int = 30,
) -> list[AnimationFrame]:
    """Build complete frame-by-frame animation.

    Args:
        visemes: Timed viseme events from phoneme pipeline.
        expression_plan: Expression events from planner.
        total_duration: Audio duration in seconds.
        fps: Target frame rate.

    Returns:
        List of AnimationFrame, one per frame.
    """
    num_frames = max(1, int(total_duration * fps))
    frames = []

    # Pre-compute blink times
    blinks = _generate_blinks(total_duration)

    # Pre-compute subtle idle head movement
    idle_seed = random.random() * 100

    for i in range(num_frames):
        t = i / fps
        w: dict[str, float] = {}

        # --- Mouth shapes from visemes ---
        mouth = _get_mouth_at_time(t, visemes)
        w.update(mouth)

        # --- Expression from plan ---
        expr_w, gesture, eyebrow = _get_expression_at_time(t, expression_plan)
        # Merge (expressions don't override mouth)
        for k, v in expr_w.items():
            if "mouth" not in k:
                w[k] = max(w.get(k, 0), v)

        # --- Blinks ---
        blink_val = _get_blink_at_time(t, blinks)
        if blink_val > 0.01:
            w["blinkLeft"] = blink_val
            w["blinkRight"] = blink_val

        # --- Eyebrow from expression ---
        if eyebrow == "raise":
            # Smooth ramp
            w["surprised"] = max(w.get("surprised", 0), 0.25)
        elif eyebrow == "flash":
            # Quick up-down on emphasis
            w["surprised"] = max(w.get("surprised", 0), 0.15)
        elif eyebrow == "furrow":
            w["angry"] = max(w.get("angry", 0), 0.15)

        # --- Head pose ---
        yaw, pitch, roll = _get_head_pose(t, total_duration, gesture, idle_seed)

        # Filter near-zero
        w = {k: round(v, 4) for k, v in w.items() if v > 0.005}

        frames.append(AnimationFrame(
            time=round(t, 4),
            weights=w,
            head_yaw=round(yaw, 3),
            head_pitch=round(pitch, 3),
            head_roll=round(roll, 3),
        ))

    return frames


def _get_mouth_at_time(t: float, visemes: list[VisemeEvent]) -> dict[str, float]:
    """Get smoothly interpolated mouth weights at time t."""
    if not visemes:
        return {}

    # Find current and next viseme
    current = None
    next_vis = None
    for i, v in enumerate(visemes):
        if v.start <= t < v.start + v.duration:
            current = v
            if i + 1 < len(visemes):
                next_vis = visemes[i + 1]
            break

    if current is None:
        # Before first or after last — rest
        return {}

    # Get weights for current viseme
    curr_w = VISEME_TO_VRM.get(current.viseme, {}).copy()

    # Intensity: stressed syllables are bigger
    intensity = 1.0
    if current.is_stressed:
        intensity = 1.2

    # Smooth transition: ease within the viseme duration
    progress = (t - current.start) / max(current.duration, 0.001)
    progress = max(0, min(1, progress))

    # Opening phase (first 40%): quick ease-out
    # Sustain (middle 30%): hold
    # Closing phase (last 30%): ease toward next viseme
    if progress < 0.4:
        # Opening — ease out (quick open)
        phase = progress / 0.4
        scale = _ease_out(phase) * intensity
    elif progress < 0.7:
        # Sustain
        scale = intensity
    else:
        # Closing — blend toward next or rest
        phase = (progress - 0.7) / 0.3
        if next_vis:
            next_w = VISEME_TO_VRM.get(next_vis.viseme, {})
            # Blend current → next
            blended = {}
            all_keys = set(curr_w.keys()) | set(next_w.keys())
            for k in all_keys:
                c = curr_w.get(k, 0) * intensity
                n = next_w.get(k, 0)
                blended[k] = c + (n - c) * _ease_in(phase)
            return blended
        else:
            scale = intensity * (1.0 - _ease_in(phase))

    return {k: v * scale for k, v in curr_w.items()}


def _get_expression_at_time(
    t: float, plan: ExpressionPlan
) -> tuple[dict[str, float], str, str]:
    """Get expression weights, gesture, and eyebrow at time t."""
    gesture = "none"
    eyebrow = "none"
    weights: dict[str, float] = {}

    for event in plan.events:
        if event.start <= t < event.start + event.duration:
            weights = expression_to_vrm_weights(event.emotion, event.intensity)
            gesture = event.head_gesture
            eyebrow = event.eyebrow

            # Fade in/out at boundaries
            progress = (t - event.start) / max(event.duration, 0.001)
            if progress < 0.1:
                fade = progress / 0.1
                weights = {k: v * fade for k, v in weights.items()}
            elif progress > 0.85:
                fade = (1.0 - progress) / 0.15
                weights = {k: v * fade for k, v in weights.items()}
            break

    return weights, gesture, eyebrow


def _get_head_pose(
    t: float, total_dur: float, gesture: str, seed: float
) -> tuple[float, float, float]:
    """Get head yaw, pitch, roll at time t."""
    # Idle subtle movement (layered sinusoids)
    yaw = (math.sin((t + seed) * 0.5) * 1.5
           + math.sin((t + seed) * 1.1) * 0.8)
    pitch = (math.sin((t + seed) * 0.3) * 0.8
             + math.sin((t + seed) * 0.8) * 0.5)
    roll = math.sin((t + seed) * 0.2) * 0.5

    # Gesture overlays
    if gesture == "nod":
        # Quick nod: pitch dip
        nod_phase = math.sin(t * 4) * 2
        pitch += nod_phase
    elif gesture == "tilt":
        # Head tilt for questions
        roll += 3.0
    elif gesture == "shake":
        # Subtle shake
        yaw += math.sin(t * 6) * 2.5

    return yaw, pitch, roll


def _generate_blinks(duration: float) -> list[tuple[float, float]]:
    """Generate natural blink times. Returns list of (time, duration) pairs."""
    blinks = []
    t = random.uniform(1.5, 3.0)
    while t < duration:
        blink_dur = random.uniform(0.1, 0.18)
        blinks.append((t, blink_dur))
        t += random.uniform(2.0, 5.0)
    return blinks


def _get_blink_at_time(t: float, blinks: list[tuple[float, float]]) -> float:
    """Get blink value (0-1) at time t."""
    for bt, bd in blinks:
        if bt <= t < bt + bd:
            progress = (t - bt) / bd
            return math.sin(progress * math.pi)  # smooth close-open
    return 0.0


def _ease_out(t: float) -> float:
    return t * (2 - t)

def _ease_in(t: float) -> float:
    return t * t
