"""Viseme definitions and viseme-to-blend-shape mapping.

Standard viseme set based on Rhubarb Lip Sync output + ARKit blend shape targets.
"""

from __future__ import annotations

# Rhubarb outputs these viseme codes
VISEME_CODES = [
    "REST",  # Silence / neutral mouth
    "A",     # "ah" — jaw wide open (pAt, fAther)
    "B",     # "ee" — lips slightly parted, wide (bEE, thE)
    "C",     # "eh" / "ah" — open mouth (bEd, bAd)
    "D",     # "oh" — rounded open mouth (bOUght, stOrm)
    "E",     # "oo" — tight round lips (nEW, blUE)
    "F",     # "f/v" — lower lip tucked under upper teeth
    "G",     # "k/g" — mouth almost closed, tongue back
    "H",     # "l" — tongue tip up, lips relaxed
    "X",     # Rest with slight parting (between words)
]

# ARKit compatible blend shape names
ARKIT_SHAPES = [
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "browDownLeft", "browDownRight",
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]

# Viseme → ARKit blend shape weights mapping
# Each viseme drives specific blend shapes to create the mouth pose
VISEME_TO_BLENDSHAPE: dict[str, dict[str, float]] = {
    "REST": {
        # Neutral — all zeros (mouth at rest)
    },
    "A": {
        # "ah" — wide open jaw
        "jawOpen": 0.7,
        "mouthLowerDownLeft": 0.3,
        "mouthLowerDownRight": 0.3,
    },
    "B": {
        # "ee" — wide smile, slight jaw open
        "jawOpen": 0.15,
        "mouthSmileLeft": 0.5,
        "mouthSmileRight": 0.5,
        "mouthStretchLeft": 0.3,
        "mouthStretchRight": 0.3,
    },
    "C": {
        # "eh"/"ah" — open relaxed
        "jawOpen": 0.45,
        "mouthLowerDownLeft": 0.2,
        "mouthLowerDownRight": 0.2,
        "mouthUpperUpLeft": 0.1,
        "mouthUpperUpRight": 0.1,
    },
    "D": {
        # "oh" — rounded open
        "jawOpen": 0.4,
        "mouthFunnel": 0.5,
        "mouthPucker": 0.3,
    },
    "E": {
        # "oo" — tight pursed lips
        "jawOpen": 0.1,
        "mouthPucker": 0.7,
        "mouthFunnel": 0.6,
    },
    "F": {
        # "f/v" — lower lip under teeth
        "jawOpen": 0.05,
        "mouthRollLower": 0.6,
        "mouthUpperUpLeft": 0.2,
        "mouthUpperUpRight": 0.2,
        "mouthClose": 0.3,
    },
    "G": {
        # "k/g" — mouth nearly closed, tension
        "jawOpen": 0.08,
        "mouthClose": 0.4,
        "mouthPressLeft": 0.3,
        "mouthPressRight": 0.3,
    },
    "H": {
        # "l" — tongue up, relaxed lips
        "jawOpen": 0.3,
        "tongueOut": 0.15,
        "mouthLowerDownLeft": 0.1,
        "mouthLowerDownRight": 0.1,
    },
    "X": {
        # Rest with slight parting (between words)
        "jawOpen": 0.03,
        "mouthClose": 0.1,
    },
}


# VRM-style blend shape mapping (for avatars with blendShape2.mouth_a etc.)
VISEME_TO_VRM: dict[str, dict[str, float]] = {
    "REST": {},
    "A": {
        "blendShape2.mouth_a": 1.0,
    },
    "B": {
        "blendShape2.mouth_i": 0.7,
        "blendShape2.mouth_e": 0.3,
    },
    "C": {
        "blendShape2.mouth_e": 0.7,
        "blendShape2.mouth_a": 0.3,
    },
    "D": {
        "blendShape2.mouth_o": 1.0,
    },
    "E": {
        "blendShape2.mouth_u": 1.0,
    },
    "F": {
        "blendShape2.mouth_i": 0.4,
        "blendShape2.mouth_u": 0.3,
    },
    "G": {
        "blendShape2.mouth_i": 0.2,
    },
    "H": {
        "blendShape2.mouth_a": 0.4,
        "blendShape2.mouth_e": 0.2,
    },
    "X": {
        "blendShape2.mouth_i": 0.05,
    },
}


def detect_blendshape_format(shape_names: list[str]) -> str:
    """Detect whether avatar uses ARKit or VRM blend shape names.

    Returns:
        "vrm" if VRM-style names detected, "arkit" otherwise.
    """
    for name in shape_names:
        if "blendShape" in name or "mouth_a" in name or "mouth_i" in name:
            return "vrm"
    return "arkit"


def get_blendshape_weights(viseme: str, format: str = "arkit") -> dict[str, float]:
    """Get blend shape weights for a given viseme code.

    Args:
        viseme: Viseme code (e.g., "A", "REST").
        format: "arkit" or "vrm" — determines which mapping to use.
    """
    if format == "vrm":
        return VISEME_TO_VRM.get(viseme, VISEME_TO_VRM["REST"]).copy()
    return VISEME_TO_BLENDSHAPE.get(viseme, VISEME_TO_BLENDSHAPE["REST"]).copy()


def interpolate_weights(
    current: dict[str, float],
    target: dict[str, float],
    factor: float,
) -> dict[str, float]:
    """Smoothly interpolate between two blend shape weight dicts.

    Args:
        current: Current blend shape weights.
        target: Target blend shape weights.
        factor: Interpolation factor (0=current, 1=target).

    Returns:
        Interpolated weights dict.
    """
    all_keys = set(current.keys()) | set(target.keys())
    result = {}
    for key in all_keys:
        cur = current.get(key, 0.0)
        tgt = target.get(key, 0.0)
        result[key] = cur + (tgt - cur) * factor
    return result
