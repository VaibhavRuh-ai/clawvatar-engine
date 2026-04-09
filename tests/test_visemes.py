"""Tests for viseme definitions and blend shape mapping."""

from clawvatar.lipsync.visemes import (
    VISEME_CODES,
    VISEME_TO_BLENDSHAPE,
    get_blendshape_weights,
    interpolate_weights,
)


def test_all_visemes_have_mapping():
    for code in VISEME_CODES:
        weights = get_blendshape_weights(code)
        assert isinstance(weights, dict)


def test_rest_is_empty():
    weights = get_blendshape_weights("REST")
    assert len(weights) == 0


def test_a_has_jaw_open():
    weights = get_blendshape_weights("A")
    assert "jawOpen" in weights
    assert weights["jawOpen"] > 0.5


def test_unknown_viseme_returns_rest():
    weights = get_blendshape_weights("UNKNOWN")
    assert len(weights) == 0


def test_interpolate_basic():
    a = {"jawOpen": 0.0}
    b = {"jawOpen": 1.0}
    result = interpolate_weights(a, b, 0.5)
    assert abs(result["jawOpen"] - 0.5) < 0.001


def test_interpolate_missing_keys():
    a = {"jawOpen": 1.0}
    b = {"mouthSmileLeft": 1.0}
    result = interpolate_weights(a, b, 0.5)
    assert abs(result["jawOpen"] - 0.5) < 0.001
    assert abs(result["mouthSmileLeft"] - 0.5) < 0.001
